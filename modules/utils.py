

def get_model_mask(args):

    if args.GAN_moment_flag:
        model_mask = "moment"
    else:
        model_mask = "Adam5"

    # sample rate
    model_mask += "client{}sr{}_clientEpoch{}".format(args.clients_number, args.clients_sample_ratio, args.clients_training_epoch)

    if args.homo_flag:
        model_mask = "homo_" + model_mask


    if args.softmax_flag:
        model_mask += "_softmaxLg"


        
    if args.resnet_flag:
        model_mask += "_rn"

        
    if args.new_VAE:
        model_mask += "_new_VAE{}_interval{}".format(args.latent_size, args.VAE_training_interval)
        
        
    if args.op:
        model_mask += "_op{}".format(args.latent_size)


    if args.noniid_flag:
        model_mask += "_noniid"

    if args.KLD_flag:
        model_mask += "_KLD"
        
    
        

    if args.asyn_FL_flag:
        model_mask += "_asynFL"

    if args.y_distillation_flag and not args.no_logits_flag:
        model_mask += "_y_logits"

    
    return model_mask



def get_tb_logs_name(args):
    
    model_mask = get_model_mask(args)
    
    if args.test:
        cmp_log_dir = "./tb_logs/debug_fedavg/{}_{}/fold{}_{}".format(args.dataset, 
                                                                      args.clients_sample_ratio,
                                                                      args.data_size_fold, 
                                                                      model_mask) 
        
    else:
        if args.cita == 1:
            if args.VAE:
                params_string = "_alpha{}beta{}_lr{}/test".format(args.alpha, args.beta, args.learning_rate)
            else:
                params_string = "_alpha{}_lr{}/test".format(args.alpha, args.learning_rate)
        else:
            params_string = "_alpha{}beta{}cita{}_lr{}/test".format(args.alpha, args.beta, args.cita, args.learning_rate)
            
        based_model = "rn"
        if args.resnet_flag:
            based_model = "rn{}_{}_{}".format(args.clients_sample_ratio, args.dataset, args.batch_size)
        elif args.old_CNN:
            based_model = "light_CNN{}_{}_{}".format(args.clients_sample_ratio, args.dataset, args.batch_size)
        else:
            based_model = "CNN{}_{}_{}".format(args.clients_sample_ratio, args.dataset, args.batch_size)
            
        # if args.enable_scheduler:
        based_model += "_schd"
            
        if args.homo_flag:
            cmp_log_dir = "./tb_logs/cmp_homo/" + based_model + "/" + model_mask + params_string
        else:
            cmp_log_dir = "./tb_logs/cmp/" + based_model + "/" + model_mask + params_string
        
    return cmp_log_dir
    
