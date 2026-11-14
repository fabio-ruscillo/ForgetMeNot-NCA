import pytest
import torch
from nca_memory.nca import LocalPerception, CellularMemory


@pytest.mark.parametrize("channels", [1, 16, 40])
def test_forward_returns_five_features_per_state_channel(channels):
    """Check that the output width, scales with the state channels."""
    cube_size = 7
    spatial_shape = (cube_size,) * 3
    state = torch.ones(1, channels, *spatial_shape)

    features = LocalPerception(channels=channels)(state)

    assert features.shape == (1, 5 * channels, *spatial_shape)


def test_depthwise_filters_keep_channels_independent():
    """Check that an impulse activates only its channel's filters."""
    channels = 3
    cube_size = 5
    spatial_shape = (cube_size,) * 3
    center = cube_size // 2

    state = torch.zeros(1, channels, *spatial_shape)
    state[0, 0, center, center, center] = 1.0

    features = LocalPerception(channels=channels)(state)

    # Check only filter responses because the input impulse is also in features.
    responses = features[0, channels:].reshape(
        channels, -1, *spatial_shape
    )
    # Check that the impulse only activates its own channel's filters.
    # Any response in an inactive channel would reveal channels mixing.
    assert torch.count_nonzero(responses[1:]) == 0

    # Check that the active channel's filters produce a nonzero response.
    active_responses = responses[0].flatten(start_dim=1)
    assert (active_responses.count_nonzero(dim=1) > 0).all()

@pytest.mark.parametrize(
    ("axis", "filter_index"),
    [(4, 0), (3, 1), (2, 2)],
    ids=["x", "y", "z"],
)
def test_sobel_unit_ramp_activates_its_axis(axis, filter_index):
    """Check the orientation, sign and scale of the Sobel filters."""
    # A unit ramp gives us a known slope of +1 along the chosen axis.
    ramp = torch.arange(5, dtype=torch.float32).reshape(1, 1, 1, 1, 5)

    # Repeat the ramp across the other axes so their slopes stay zero.
    state = ramp.movedim(-1, axis).expand(1, 1, 5, 5, 5)

    features = LocalPerception(channels=1)(state)

    # Compare only the Sobel responses from the interior cells, because the
    # ramp jumps from 4 to 0 at the periodic boundary.
    sobel = features[:, 1:4, 1:-1, 1:-1, 1:-1]

    # The normalized filter should recover +1 along the ramp and zero
    # along the two perpendicular directions.
    expected = torch.zeros_like(sobel)
    expected[:, filter_index] = 1.0

    # Allow a small rounding error, including for the expected zeros.
    torch.testing.assert_close(sobel, expected, rtol=0, atol=1e-6)


def test_laplacian_of_constant_state_is_zero_everywhere():
    """Check that a constant state has zero Laplacian, including at edges."""
    cube_size = 5
    spatial_shape = (cube_size,) * 3

    # A constant state makes the neighbor mean equal to the center value.
    state = torch.ones(1, 1, *spatial_shape)
    features = LocalPerception(channels=1)(state)

    laplacian = features[0, -1]

    torch.testing.assert_close(
        laplacian, torch.zeros_like(laplacian), rtol=0, atol=1e-6
    )


@pytest.fixture
def perception_gradient():
    """Compute input gradients from the filter responses."""
    # Initialized both channels so each has a signal to backpropagate.
    state = torch.zeros(1, 2, 5, 5, 5)
    state[0, 0, 2, 2, 2] = 1.0
    state[0, 1, 2, 2, 2] = 2.0

    state.requires_grad_()

    # Exclude the copied state from the gradient check.
    # Including it could give a nonzero gradient even if the filter responses
    # were detached from the computation graph.
    responses = LocalPerception(channels=2)(state)[:, 2:]

    # Square the responses so positive and negative values cannot cancel.
    return torch.autograd.grad(responses.square().sum(), state)[0]


def test_filter_responses_produce_finite_gradients(perception_gradient):
    """Check for NaN or infinite gradients in the perception path."""
    # A single non finite component could make a training update invalid.
    assert torch.isfinite(perception_gradient).all()


def test_filter_responses_produce_nonzero_gradients(perception_gradient):
    """Check that each active channel gets a learning signal."""
    # Combine spatial positions to check the learning signal per channel.
    assert (perception_gradient.flatten(2).count_nonzero(dim=2) > 0).all()


def test_batched_forward_matches_separate_forwards():
    """Check that batching does not change any story's features."""
    perception = LocalPerception(channels=3)

    # Use distinct inputs to expose mixing.
    generator = torch.Generator().manual_seed(0)
    state = torch.randn(2, 3, 5, 5, 5, generator=generator)
    
    batched = perception(state)

    # Use single cube forwards as a reference for independent processing.
    # Keep a batch axis of size one so each input fits LocalPerception.
    separate = torch.cat(
        [perception(sample) for sample in state.split(1)], dim=0
    )

    # Check that the results are the same, no matter the type of operations.
    torch.testing.assert_close(batched, separate, rtol=1e-5, atol=1e-6)


def test_cellular_memory_update_rule_receives_gradients():
    """Check that both layers receive finite, nonzero gradients."""
    batch_size = 2
    channels = 2
    cube_size = 5

    # Save and restore the CPU random state so this test does not change
    # the random sequence used by other tests. This test uses only the CPU.
    with torch.random.fork_rng(devices=[]):
        # Fix the CPU seed before creating both the model and the input.
        # This reproduces the initial weights as well as the two cubes.
        torch.default_generator.manual_seed(0)
        memory = CellularMemory(channels=channels)
        state = torch.randn(
            batch_size, channels, cube_size, cube_size, cube_size
        )

    # Use a scalar function of the updated state to check the gradient path
    # through one memory update before integrating the predictor.
    test_objective = memory(state).square().mean()
    gradients = torch.autograd.grad(
        test_objective, memory.update_rule.parameters()
    )

    for gradient in gradients:
        # A NaN or infinity could corrupt the corresponding weight update.
        assert torch.isfinite(gradient).all()

        # Each layer must get a signal on at least one weight. Individual
        # weights can have zero gradient, for example when ReLU is inactive.
        assert torch.count_nonzero(gradient) > 0