import numpy as np


def policy_based_run(agent, optimizer, n_episodes=1000, t_max=1000, gamma=1.0):
    assert hasattr(optimizer, '__class__'), 'optimizer should be of type class.'
    assert hasattr(optimizer, "step"), 'optimizer should have `step` method for optimization.'
    assert hasattr(agent, '__class__'), 'agent should of type class.'

    PRINT_EVERY = 100
    TARGET_SCORE = 500.0
    scores = []
    Ps = [agent.model.get_params_array()]  # get the model params as the x0.

    for episode in range(1, n_episodes + 1):

        # calculate discounted return on a complete episode
        Gs = [agent.calculate_return(params=p, gamma=gamma, t_max=t_max) for p in Ps]
        Ps = optimizer.step(Ps, Gs)

        scores.append(max(Gs))

        print(f'Episode {episode:03}/{n_episodes}: Score={scores[-1]}', end='\r')
        if episode % PRINT_EVERY == 0:
            S = np.mean(scores[-PRINT_EVERY:])
            print(f'Episode {episode:03}/{n_episodes}: Average Score for last 100 episodes={S}')

        if np.mean(scores[-100:]) >= TARGET_SCORE:
            print(f'Agent reached the target score of {TARGET_SCORE} in {episode} episodes!')
            break

    return scores
