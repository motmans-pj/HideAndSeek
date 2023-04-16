import os, sys
import numpy as np
from statistics import mean

def game_train(env,seeker,hider, n_ep,verbose = True):
  n_steps = []
  for ep in range(1,n_ep+1):
        n_steps_ep = 0
        obs = env.reset()
        
        if verbose:
          if ep % 1000 == 0:
            print("\rEpisode {}/{}.".format(ep,n_ep), end="")
            sys.stdout.flush()
            # Print average of last 1000 episodes
            print(f'\n The average number of steps is: {np.mean(n_steps)}')
        
          n_steps = []
        
        while True: 
          av_actions = env.f_available_actions()
          #if len(av_actions['Hider']) == 1:
           # env.render()

          # Epsilon-greedy strategy on the available agents
          action_seeker = seeker.policy(obs,av_actions['Seeker'])
          action_hider = hider.policy(obs,av_actions['Hider'])
          env.actions = {'Seeker':action_seeker,'Hider':action_hider}



          # Apply action and return new observation of the environment
          new_obs, rewards, terminations,truncations,info = env.step(env.actions)
          # Update the Q-values
          seeker.update(obs['Seeker'],action_seeker,rewards["Seeker"],terminations["Seeker"],new_obs['Seeker'])
          hider.update(obs['Hider'],action_hider,rewards["Hider"],terminations["Hider"],new_obs['Hider'])
          obs = new_obs

          n_steps_ep +=1 


          
          if truncations["Seeker"]:
            #print('Too many steps, game over')
            n_steps.append(n_steps_ep)
            break
          if terminations["Hider"]:
            #print("The Seeker Won!",n_steps)
            n_steps.append(n_steps_ep)
            break

        seeker.decay_epsilon()
        hider.decay_epsilon()


def game_test(env,seeker,hider, n_ep_train, n_ep_test, static = False):
  
  game_train(env,seeker,hider, n_ep_train)
  n_hider_victories = 0
  n_seeker_victories = 0
  n_steps_to_vistory = []
  for ep in range(1,n_ep_test+1):
      n_steps = 0
      obs = env.reset()

      while True: 
        av_actions = env.f_available_actions()
        #if len(av_actions['Hider']) == 1:
          # env.render()
        action_seeker = seeker.policy(obs,av_actions['Seeker'])
        action_hider = hider.policy(obs,av_actions['Hider'])
        env.actions = {'Seeker':action_seeker,'Hider':action_hider}
        new_obs, rewards, terminations,truncations,info = env.step(env.actions)
        obs = new_obs
        n_steps+=1

        if ep == n_ep_test -1:
            env.render_rgb()

        if truncations["Seeker"]:
            #print('Too many steps, game over')
            n_hider_victories+=1
            n_steps_to_vistory.append(n_steps)
            break
        if terminations["Hider"]:
            #print("The Seeker Won!",n_steps)
            n_seeker_victories+=1
            n_steps_to_vistory.append(n_steps)
            break
        

  print(f'The hider has won {n_hider_victories} times, \nThe seeker has won {n_seeker_victories} times\n')
  if static:
    return n_steps_to_vistory



def avg_victories(envir, seeker,hider, n_episodes,n_series):
  results = []
  for s in range(n_series):
    n_hider_victories = 0
    n_seeker_victories = 0
    for ep in range(1,n_episodes+1):
        obs = envir.reset()

        while True:
          av_actions = envir.f_available_actions()
          #if len(av_actions['Hider']) == 1:
            # env.render()
          action_seeker = seeker.policy(obs,av_actions['Seeker'])
          action_hider = hider.policy(obs,av_actions['Hider'])
          envir.actions = {'Seeker':action_seeker,'Hider':action_hider}
          new_obs, rewards, terminations,truncations,info = envir.step(envir.actions)
          obs = new_obs

          if truncations["Seeker"]:
              #print('Too many steps, game over')
              n_hider_victories+=1
              #n_steps_to_vistory.append(n_steps)
              break
          if terminations["Hider"]:
              #print("The Seeker Won!",n_steps)
              n_seeker_victories+=1
              #n_steps_to_vistory.append(n_steps)
              break
    results.append(n_hider_victories)

  return mean(results)
