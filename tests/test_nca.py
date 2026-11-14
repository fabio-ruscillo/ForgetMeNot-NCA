import pytest
import torch
from nca_memory.nca import LocalPerception, CellularMemory, write_center


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

    generator = torch.Generator(device=state.device).manual_seed(0)

    # Use a scalar function of the updated state to check the gradient path
    # through one memory update before integrating the predictor.
    test_objective = memory(state, generator=generator).square().mean()
    gradients = torch.autograd.grad(
        test_objective, memory.update_rule.parameters()
    )

    for gradient in gradients:
        # A NaN or infinity could corrupt the corresponding weight update.
        assert torch.isfinite(gradient).all()

        # Each layer must get a signal on at least one weight. Individual
        # weights can have zero gradient, for example when ReLU is inactive.
        assert torch.count_nonzero(gradient) > 0


def test_stochastic_update_masks_whole_cells():
    """Check that each cell updates all its channels or keeps its state."""
    cube_size = 3
    mask_seed = 0

    # Use the same weights and input every time this test runs.
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(0)
        memory = CellularMemory(channels=2)
        state = torch.randn(1, 2, cube_size, cube_size, cube_size)

    # With the default probability of 1.0, is able to see how every cell
    # would look like if it updated.
    generator = torch.Generator().manual_seed(mask_seed)
    full_update = memory(state, generator=generator)

    # Run the stochastic update with the same weights and input.
    memory.update_probability = 0.5
    actual = memory(state, generator=generator)

    # Reset the seed to get the same random numbers as the forward pass.
    generator.manual_seed(mask_seed)
    selected = torch.rand(state[:, :1].shape, generator=generator) < 0.5

    # Selected cells should match the full update. Skipped cells should
    # keep their original state. This must hold for both channels.
    expected = torch.where(selected, full_update, state)
    torch.testing.assert_close(actual, expected)


def test_evolve_matches_two_consecutive_updates():
    """Check that each step uses the state produced by the previous one."""
    # Reproduce the weights and input without affecting other tests.
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(0)
        memory = CellularMemory(channels=2, update_probability=0.5)
        state = torch.randn(1, 2, 3, 3, 3)

    # Start both runs from the same seed so they draw the same masks.
    generator = torch.Generator(device=state.device).manual_seed(0)
    reference_generator = torch.Generator(device=state.device).manual_seed(0)

    actual = memory.evolve(state, steps=2, generator=generator)

    # Use the first result as input so the second step builds on it.
    first_step = memory(state, generator=reference_generator)
    expected = memory(first_step, generator=reference_generator)

    torch.testing.assert_close(actual, expected)


def test_write_center_writes_feedback_to_a_copy():
    """Check that feedback is written while the input memory stays intact."""
    state = torch.ones(1, 2, 3, 3, 3)
    feedback = torch.zeros(1, 2)
    original = state.clone()

    # Keep the state before the feedback intact for backpropagation.
    updated_state = write_center(state, feedback)

    torch.testing.assert_close(
        updated_state[:, :, 1, 1, 1], feedback, rtol=0, atol=0
    )
    # The earlier prediction may still need the original state for backward.
    torch.testing.assert_close(state, original, rtol=0, atol=0)