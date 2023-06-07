
import cv2
import random
import numpy as np
import argparse
from DRL.evaluator import Evaluator
from utils.util import *
from utils.tensorboard import TensorBoard
import time
import datetime
from options.options import Options
from DRL.feedback import HumanFeedback
from DRL.init_stroke import SaliencyMap
#import tqdm
from tqdm import tqdm

date_and_time = datetime.datetime.now()
run_name = 'painter_' + date_and_time.strftime("%m_%d__%H_%M_%S")
# writer = TensorBoard('train_log_cats/{}'.format(run_name))
writer = TensorBoard('train_log_deleteme/{}'.format(run_name))

if not os.path.exists('models/DRL'):
    os.mkdir('models/DRL')

def train(agent, env, evaluate):
    train_times = opt.train_times
    env_batch = opt.env_batch
    validate_interval = opt.validate_interval
    max_step = opt.max_step
    debug = opt.debug
    episode_train_times = opt.episode_train_times
    resume = opt.resume
    output = opt.output
    time_stamp = time.time()
    step = episode = episode_steps = 0
    tot_reward = 0.
    observation = None
    noise_factor = opt.noise_factor
    imgs_used_from_file = 0
    feedback = opt.feedback
    if feedback:
        feedback_interval = opt.feedback_interval
        feedback_type = opt.feedback_type
        feedback_max = opt.feedback_max

    pbar = tqdm(total=train_times)
    while step <= train_times:
        step += 1
        episode_steps += 1
        # reset if it is the start of episode
        if observation is None:
            observation = env.reset()[0]

            agent.reset(observation, noise_factor)

            imgs_used_from_file += agent.batch_size
            if (env.dataset == 'all' and imgs_used_from_file > env.env.train_num):
                env.env.load_new_file()
                print('loading a new file')
                imgs_used_from_file = 0



        # Collect data from sampling
        if step < opt.warmup or step % 100 == 0:
            print("FROM SAMPLING", observation.shape)
            #env batchsize 1 for now
            n_strokes = 5 # TODO: param manual in ddpg.py 
            smap = SaliencyMap(observation[0, 3:6].unsqueeze(0).float(), n_strokes = 5)
            indices = smap.inds_normalized
            print(indices)

            action = torch.zeros((1,n_strokes*10))#.to("cuda") # batch, n strokes * 10
            for i in range(n_strokes):
                _action = torch.zeros(10) 
                # data
                _action[0] = random.random() # float(stroke_length)
                _action[1] = random.random() # float(stroke_bend)
                _action[2] = 0.25 #random.random() # float(stroke thickness)
                _action[3] = 0.0 #float(stroke alpha)
                
                _action[4] = random.random() #float(input("rotation 0-1: ")) #rotation 
                _action[5] = indices[i][0]
                _action[6] = indices[i][1] 
                
                _action[7] = 1#color_r 
                _action[8] = 1#color_g 
                _action[9] = 1#color_b 
                action[0][i*10:i*10+10]=_action
                
            agent.set_action(observation, action)

        else:
            #prev_observation = agent.state
            action = agent.select_action(observation, noise_factor=noise_factor) # updates agent.action
            #print("Action Shape:", action.shape)
            #observation, reward, done, _, mask = env.step(action, episode_steps) # updates env.canvas, env.stepnum
            
            ## Add attention Map

        #TODO: debug gray stroke in the middle
        #TODO: decide frequency of feedback
        #print(step, episode, episode_steps)
    
        if feedback and episode <= feedback_max and episode % feedback_interval == 0 and episode_steps <= 5:
            print("Entering feedback for episode ", episode, "and step ", step)

            # compute corrected action
            prev_action = action
            prev_observation = agent.state
            #next_observation = observation
            action_corrected = HumanFeedback.select_new_action(prev_action, prev_observation) #, next_observation)

            # get new observation and reward            
            observation_corrected, reward_corrected, done_corrected, _, mask_corrected = env.step(action_corrected, episode_steps)

            # store in buffer
            #agent.state = prev_observation
            agent.state = observation_corrected
            agent.action = action_corrected
            agent.observe(reward_corrected, observation_corrected, done_corrected, step, mask_corrected)
        else: 
            observation, reward, done, _, mask = env.step(action, episode_steps) # updates env.canvas, env.stepnum
            
            agent.observe(reward, observation, done, step, mask) #updates agent.state, agent.memory


        if (episode_steps >= max_step and max_step):

            if step > opt.warmup:
                # [optional] evaluate
                #if episode > 0 and validate_interval > 0 and episode % validate_interval == 0:
                #print("1", episode % validate_interval)
                if validate_interval > 0 and episode % validate_interval == 0:
                    reward, dist = evaluate(env, agent.select_action, debug=debug)
                    #env.save_image(step, episode_steps)
                    
                    if debug: print('Step_{:07d}: mean_reward:{:.3f} mean_dist:{:.3f} var_dist:{:.3f}'\
                        .format(step - 1, np.mean(reward), np.mean(dist), np.var(dist)))
                    writer.add_scalar('validate/mean_reward', np.mean(reward), step)
                    writer.add_scalar('validate/mean_dist', np.mean(dist), step)
                    writer.add_scalar('validate/var_dist', np.var(dist), step)
                    agent.save_model(output)
            train_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            tot_Q = 0.
            tot_value_loss = 0.
            if step > opt.warmup:
                #if step < 10000 * max_step:
                #    lr = (9e-4, 3e-3)
                #elif step < 20000 * max_step:
                #    lr = (3e-4, 9e-4)
                #else:
                lr = (3e-7, 1e-6)
                 
                #     lr = (9e-5, 3e-4)
                # lr = (3e-6, 1e-5)
                # if step < 10000 * max_step:
                #     lr = (3e-4, 1e-3)
                # elif step < 20000 * max_step:
                #     lr = (1e-4, 3e-4)
                # else:
                #     lr = (3e-5, 1e-4)

                for i in range(episode_train_times):
                    Q, value_loss = agent.update_policy(lr, episode_steps)
                    tot_Q += Q.data.cpu().numpy()
                    tot_value_loss += value_loss.data.cpu().numpy()
                
                writer.add_scalar('train/critic_lr', lr[0], step)
                writer.add_scalar('train/actor_lr', lr[1], step)
                writer.add_scalar('train/Q', tot_Q / episode_train_times, step)
                writer.add_scalar('train/critic_loss', tot_value_loss / episode_train_times, step)
            if debug: print('#{}: steps:{} interval_time:{:.2f} train_time:{:.2f}' \
                .format(episode, step, train_time_interval, time.time()-time_stamp)) 
            time_stamp = time.time()
            # reset
            observation = None
            episode_steps = 0
            episode += 1
        pbar.update(1)
    pbar.close()



if __name__ == "__main__":
    opt = Options().parse()

    opt.output = get_output_folder(opt.output, "Paint")
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(opt.seed)
    random.seed(opt.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    from DRL.ddpg import DDPG
    from DRL.multi import fastenv
    # fenv = fastenv(args.max_step, args.env_batch, writer, args.canvas_color, args.loss_fcn, args.dataset, args.use_multiple_renderers)
    # agent = DDPG(args.batch_size, args.env_batch, args.max_step, \
    #              args.tau, args.discount, args.rmsize, \
    #              writer, args.resume, args.output, args.loss_fcn, args.renderer, args.use_multiple_renderers)
    fenv = fastenv(opt, writer)
    agent = DDPG(opt, writer)
    evaluate = Evaluator(opt, writer)
    print('observation_space', fenv.observation_space, 'action_space', fenv.action_space)

    summary = 'Loss Function - {}\nRenderer - {}\nResuming Model - {}\nbatch_size - {}\nmax_step - {}\nOutput - {}' \
        .format(opt.loss_fcn, opt.renderer, opt.resume, opt.batch_size, opt.max_step, opt.output)
    writer.add_text('summary', summary, 0)
    writer.add_text('Command Line Arguments', str(opt), 0)

    train(agent, fenv, evaluate)
