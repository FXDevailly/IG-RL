import os 
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from functools import partial
import traci
from time import time
import matplotlib.pyplot as plt
import random
import torch 
from utils.comput_process import *
from utils.train_process import * # train_process
from utils.gen_model import * # gen_process
import numpy as np
from utils.experiment_process import Experiment
from IPython.display import clear_output
from utils.env_classes import *
import collections
from sumolib import checkBinary
import argparse
import configparser
import signal
from subprocess import DEVNULL, STDOUT, check_call
import xml.etree.ElementTree as ET
class TimeoutException(Exception):   # Custom exception class
    pass
def timeout_handler(signum, frame):   # Custom signal handler
    raise TimeoutException
# Change the behavior of SIGALRM
signal.signal(signal.SIGALRM, timeout_handler)
    




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-dir', type=str, nargs='+', required=True, help="location of configurations")
    args = parser.parse_args()
    return args

def create_exp(params, seed, real_net, real_net_address, n_workers, memory_queue,  request_end, learn_model_queue, comput_model_queue, reward_queue, additional_reward_queue, Policy_Type, mode, greedy_reward_queue, tested_end, tested, num):

############################################################################################################################################

    def get_network():
        netaddress = 'sumo_nets/' + str(seed) +'/myNet.net.xml'
        os.system("rm -r " + folderaddress)
        os.system("mkdir " + folderaddress)
        os.system("cp sumo_nets/" + ("myConfigGen.sumocfg" if additional_env_params['gen_trips_before_exp'] else "myConfig.sumocfg")+ " " + folderaddress)
        if real_net :
            if additional_env_params['net_name'] == 'Manhattan':
                os.system("cp " + real_net_address + " " + folderaddress + "/myNet.net.xml" +  ' > /dev/null 2>&1')
            else:
                os.system("netconvert --osm-files " + real_net_address + " -o " + folderaddress +"/myNet.net.xml " + "--no-turnarounds --no-internal-links" + ' > /dev/null 2>&1')
        else:
            if additional_env_params["grid"] == True:
                os.system("netgenerate --grid --seed=" + (str(seed) if (mode == 'train' and not additional_env_params["specialist"] and additional_env_params["GCN"]) else str(100)) + " -o " + netaddress + ' --grid.length=' + str(additional_env_params["grid_lane_length"]) + " --grid.attach-length=" + str(additional_env_params["grid_lane_length"]) + " --grid.x-number=" + str(additional_env_params["col_num"]) + " --grid.y-number=" + str(additional_env_params["row_num"]) + " -j=traffic_light --no-turnarounds=true" + ' > /dev/null 2>&1' )
            else:
                os.system("netgenerate --rand --default.lanenumber=" + str(additional_env_params["max_num_lanes"]) + " --rand.random-lanenumber --seed=" + (str(int(seed+(num*n_workers))) if (not additional_env_params["specialist"] and additional_env_params["GCN"]) else str(int(2000+(num if num!=2 else 5)))) + " -o " + netaddress + " --rand.iterations=" + str(additional_env_params['num_edges_random_net_train']) + " -j=traffic_light --no-turnarounds=true --rand.max-distance=" + str(additional_env_params['max_lane_length']) + ' --rand.min-distance=' + str(additional_env_params['min_lane_length'])  + ' > /dev/null 2>&1')

        return netaddress

    def gen_traffic():

        os.system('rm '+ folderaddress + "/trips.rou.xml" + ' > /dev/null 2>&1')
        os.system('rm '+ folderaddress + "/trips.trips.xml" + ' > /dev/null 2>&1')
        os.system("python utils/randTrips.py -n " + folderaddress +"/myNet.net.xml --demand-duration " +str(additional_env_params["demand_duration"]) + " --lane-demand-variance " +str(additional_env_params["lane_demand_variance"]) +" --min-distance "+ str(additional_env_params["min_distance"]) +" --fringe-factor " +str(additional_env_params["fringe_factor"]) + " --seed " + str(int(seed + 1 + (n_workers*num) + (200 if mode == 'test' else (step_counter+1)) + (10000 if (num == 3 and seed == 10 and additional_env_params["period"] == 2) else 0))) + " -b 0 -e "+ str(additional_env_params["nb_steps_per_exp"] * (1 if (not real_net and mode == 'test') else 1))+" --binomial 100 --period " + str(additional_env_params["period"]) + " --validate true -o " + folderaddress + "/trips.rou.xml -r " + folderaddress + "/trips.trips.xml" + ' > /dev/null 2>&1')


        tree = ET.parse(folderaddress + '/trips.rou.xml')
        root = tree.getroot()
        trips_dict = collections.OrderedDict()
        for child in root:
            if int(float(child.attrib['depart'])) < 3600:
                trips_dict[child.attrib['id']] = collections.OrderedDict({key:value for key,value in child.attrib.items() if key != 'id'})
        return trips_dict
    def gen_folders():
        additional_env_params['tb_foldername']+=  '/' + (str(num) if mode == 'train' else (str(tested.split("_")[0]) + "_" + str(num)))
        os.system("rm -r " + additional_env_params["tb_foldername"] + ' > /dev/null 2>&1') 
        os.system("mkdir " + additional_env_params["tb_foldername"] + ' > /dev/null 2>&1')
        if mode == 'train':
            additional_env_params['save_model_path'] +=  '/' + str(num) 
            os.system("mkdir " + additional_env_params["save_model_path"] + ' > /dev/null 2>&1')
        else:
            additional_env_params['load_model_path'] +=  '/' + str(num) + '/' + "params_checkpoint.pt"           
            additional_env_params['rendering_path'] +=  '/' + str(tested) 
            os.system("mkdir " + additional_env_params["rendering_path"] + ' > /dev/null 2>&1')
            additional_env_params['rendering_path'] +=  '/' +  str(num)
            os.system("rm -r " + additional_env_params["rendering_path"] + ' > /dev/null 2>&1') 
            os.system("mkdir " + additional_env_params["rendering_path"] + ' > /dev/null 2>&1')
############################################################################################################################################
        

    
    
    # TEST MULTIPLE MODELS IN A SINGLE RUN (6 seeds per run)
    if mode == 'test' and real_net:
        num += seed//(n_workers/5)    
        num = int(num)
        print(seed, ' : ', num)
        
        
        
    params['gen_trips_before_exp'] = True
    params['max_time_between_actions'] = 60
    
    if params['policy'] == 'binary':
        params['generated_graphs'] += ['tl_connection_lane_graph']
        params['graph_of_interest'] = 'tl_connection_lane_graph'
    else:
        params['generated_graphs'] += ['full_graph']   
        params['graph_of_interest'] = 'full_graph'   
        
    
    if params['render']:   
        params['render'] = True
    else:
        params['render'] = False
        params['save_render'] = False
        
    if mode == 'test':
        params['EPS_START'] = 0
        
        
    params['num_specialist'] = n_workers * params['specialist']
    if seed < int(params['num_specialist']):
        params['specialist'] = True
    else:
        params['specialist'] = False

    additional_env_params = params
    

    
    
    
    
    
    if torch.cuda.is_available():
        train_device = 'cuda:0'
        if torch.cuda.device_count() == 1:
            comput_device = 'cuda:0'
        else:
            comput_device = 'cuda:1'
    else:
        train_device = 'cpu'
        train_device = 'cpu'  
        
    gen_folders()
        

    # START OF THE LOOP 
    signal.alarm(params['exp_real_duration'])  
    
    port = 10000 + seed + n_workers*num
    try:    
        epoch = 0
        step_counter = 0
        while True :
            if real_net:
                folderaddress = 'sumo_nets/Real/' + real_net_address.split(".")[0] + str(seed)
            else:
                folderaddress = 'sumo_nets/' + str(seed) 
                
            netaddress = get_network()
            if additional_env_params['gen_trips_before_exp'] :
                trips_dict = gen_traffic()
                    
            if additional_env_params['render']:  
                sumoBinary = checkBinary("sumo-gui")
            else:       
                sumoBinary = checkBinary("sumo")
                
                
                
            if mode == 'train':
                print("step counter :", step_counter)
                
                
            sumoCmd = [sumoBinary, "-c", folderaddress + "/" + ("myConfigGen.sumocfg" if additional_env_params['gen_trips_before_exp'] else "myConfig.sumocfg"), "--no-step-log", "--time-to-teleport=-1"]
            
            while True:
                try:
                    traci.start(sumoCmd, port = (port), label = str(port)) 
                    break
                except:
                    port +=n_workers
                    traci.close(False)

            traci_connection = traci.getConnection(str(port)) 
            env = Env(additional_env_params, traci_connection)
            env.seed = seed
            exp = Experiment(env,seed,n_workers)
            step_counter += exp.run(trips_dict, epoch, seed, num, n_workers, memory_queue,  request_end, learn_model_queue, comput_model_queue, reward_queue, additional_reward_queue, Policy_Type, mode, greedy_reward_queue, tested_end, tested)

            
            
            epoch+=1
            traci.close(False)
            traci_connection.close(False)
            del env, trips_dict, exp
            
            if mode == 'test' or step_counter >= params['exp_sim_duration']:
                signal.alarm(0)
                break                
    
    except TimeoutException as e:
        traci_connection.close(False)
        traci.close(False)
        signal.alarm(0)
        pass
    

        
        
        
        
        
        
        
def run_experiment(params, num = 0):
    mode = params['mode']
    real_net = params['real_net']
    real_net_address = params['real_net_address']
    Policy_Type = params['Policy_Type']
    tested = params['tested']    
    from pathos.helpers import mp as multiprocess
    import collections
    import random
    import torch 
    from pathos.multiprocessing import ProcessPool as Pool
    import time
    
    # INITIALIZING SEEDS 
    torch.manual_seed(0)
    greedy_training = False
    assert mode in ["train", "test"], "Invalid mode."
    assert Policy_Type in ["Q_Learning","Actor_Critic", "Critic"],"Policy is not recognized."

    if Policy_Type == "Q_Learning":
        n_workers = multiprocess.cpu_count()-2
    elif 'critic' in Policy_Type.lower() and 'actor' in Policy_Type.lower():
        n_workers = multiprocess.cpu_count()-2   

    work_processes = collections.OrderedDict()

    # QUEUES
    manager = multiprocess.Manager()
    baseline_reward_queue = manager.Queue()
    greedy_reward_queue = manager.Queue()
    learn_model_queue = manager.Queue(1) 
    comput_model_queue = manager.Queue(1)

    # DICTS
    memory_queues = collections.OrderedDict() 
    reward_queues = collections.OrderedDict() 
    request_workers_ends = collections.OrderedDict() 
    request_comput_ends = collections.OrderedDict() 

    reward_workers_ends = collections.OrderedDict() 
    reward_other_ends = collections.OrderedDict() 
    memory_workers_ends = collections.OrderedDict() 
    memory_other_ends = collections.OrderedDict() 
    baselines_workers_ends = collections.OrderedDict()
    baselines_learner_ends = collections.OrderedDict()
    tested = params['tested']
    if mode == "test":
        n_workers = len(tested) 
        tested_workers_ends = collections.OrderedDict()
        tested_learner_ends = collections.OrderedDict()

    seeds = np.asarray(list(range(n_workers))) # RANDOM SEED FOR EVERY ENVz
    params['n_workers'] = n_workers
    
    
    for idx in range(n_workers):
        if mode == "test":
            tested_workers_ends[idx], tested_learner_ends[idx] = multiprocess.Pipe()
        request_workers_ends[idx], request_comput_ends[idx] = multiprocess.Pipe()
        reward_workers_ends[idx], reward_other_ends[idx] = multiprocess.Pipe()
        memory_workers_ends[idx], memory_other_ends[idx] = multiprocess.Pipe()
        memory_queues[idx] = manager.Queue()
        reward_queues[idx] = manager.Queue()
        work_processes[idx] = multiprocess.Process(target=create_exp, args=(params,
                                                                            seeds[idx], 
                                                                            real_net,
                                                                            real_net_address,
                                                                            n_workers,
                                                                            memory_workers_ends[idx],  
                                                                            request_workers_ends[idx],
                                                                            learn_model_queue, 
                                                                            comput_model_queue,
                                                                            reward_workers_ends[idx],
                                                                            baseline_reward_queue if (mode == 'test' and idx == 1) else None,
                                                                            Policy_Type,
                                                                            mode,
                                                                            greedy_reward_queue if (idx == 0 and (greedy_training or mode == 'test')) else None,
                                                                            tested_workers_ends[idx] if mode =='test' else None,
                                                                            tested[idx] if mode =='test' else None,
                                                                            num
                                                                            ))

    # WITH Q_LEARNING, WE USE A PROCESS FOR COMPUTATION IN EXPERIENCE GENERATION, AND ANOTHER TO TRAIN
    if Policy_Type == "Q_Learning":
        learn_process =  multiprocess.Process(target=train, args=(memory_other_ends,
                                                                  learn_model_queue,
                                                                  comput_model_queue,
                                                                  reward_other_ends,
                                                                  baseline_reward_queue if mode == 'test' else None,
                                                                  greedy_reward_queue if (greedy_training or mode == 'test') else None,
                                                                  tested_learner_ends if mode == 'test' else None,
                                                                  tested if mode == 'test' else None,
                                                                  num))
        comput_process =  multiprocess.Process(target=comput, args=(request_comput_ends, comput_model_queue))
        learn_process.start()

    comput_process.start()


    for k, work_process in work_processes.items():
        work_process.start()
    for k, work_process in work_processes.items():
        work_process.join()
        work_process.terminate()

    
    comput_process.terminate()
    if Policy_Type == "Q_Learning":
        learn_process.terminate()

        

        
        
if __name__ == '__main__':
    
    # GATHERS ADRESSES OF THE 'CONFIG' FILES WHICH ARE TO BE USED 
    args = parse_args()
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(args.config_dir)    
    params = collections.OrderedDict()
    for name, section in config._sections.items():
        for k,v in section.items():
            params[k] = eval(v)
            
    # PRINTS ADRESSES OF ALL THE 'CONFIG' FILES WHICH ARE TO BE USED AND ASKS THE USER TO CONFIRM
    print(params['config_dirs'])
    go = input ('confirm run ? (y/n) :')
    if go.lower() != 'y':
        sys.exit()
        
        
    # WRITES THE ADRESSES OF ALL THE 'CONFIG' FILES WHICH ARE TO BE USED DURING THE CURRENT RUN.
    outfile = open('current_schedule.pkl','wb')
    pickle.dump(params['config_dirs'],outfile)
    outfile.close()             

    
    # ITERATE OVER ALL 'CONFIG' FILES AND THEIR CORRESPONDING PARAMETERS
    for file in params['config_dirs']:
        assert os.path.isfile(file) , "File " + file + " was not found..."
    for file in params['config_dirs']:
        config = configparser.ConfigParser()
        config.optionxform = str
        config.read(file)
        params = collections.OrderedDict()
        for name, section in config._sections.items():
            for k,v in section.items():
                params[k] = eval(v)
                
        # CREATE PATHS IN ORDER TO EASE THE ORGANIZATION OF FOLDERS
        params['file'] = file[:file.rindex('/')+1] + 'results/'
        os.system("mkdir " + params["file"] + ' > /dev/null 2>&1')           
        params['file'] += params['exp_name'] + '/'
        os.system("mkdir " + params["file"] + ' > /dev/null 2>&1')          
        params['save_model_path'] = params['file'] + 'models_params/'
        params['tb_foldername'] = params['file'] + 'tensorboard/' 
        if params['mode'] == 'train':
            params['save_model_path'] = params['file'] + 'models_params/'
            os.system("mkdir " + params["save_model_path"] + ' > /dev/null 2>&1')            
        else:
            params['rendering_path'] = params['file'] + 'rendering/' 
            os.system("rm -r " + params["rendering_path"] + ' > /dev/null 2>&1')
            os.system("mkdir " + params["rendering_path"] + ' > /dev/null 2>&1')
            
        os.system("mkdir " + params["tb_foldername"] + ' > /dev/null 2>&1')            
        os.system("rm -r " + params["tb_foldername"] + ' > /dev/null 2>&1')
        os.system("mkdir " + params["tb_foldername"] + ' > /dev/null 2>&1')

        
        
        
        # IF EVALUATING
        if params['mode'] == 'test':
            # ITERATE OVER ALL POLICIES
            for tested_policy in params['tests']:  
                params['tested']= []
                # FOR A GIVEN RUN/EXPERIMENT, WE EVALUATE EVERY POLICY USING 'N_TESTS' PARALLEL ENVIRONMENTS (e.g. 30)
                for t in range(params['n_tests']):
                    params['tested'].append(tested_policy + '_' + str(t))
                # WE RUN THE EXPERIMENT 'N_EXP' TIMES (e.g. 5)
                for n in range(params['n_exp']):   
                    run_experiment(params,n)
                    print("FINISHED EXPERIMENT :", n)

        # IF TRAINING
        else:
            params['tested'] = []
            # WE RUN THE EXPERIMENT 'N_EXP' TIMES (e.g. 5)
            for n in range(params['n_exp']):  
                run_experiment(params,n)
            print("FINISHED EXPERIMENT :", n)
    sys.exit()

