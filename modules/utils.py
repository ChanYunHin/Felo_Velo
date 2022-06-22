

def get_model_mask(args):

    if args.GAN_moment_flag:
        model_mask = "moment"
    else:
        model_mask = "Adam"

    # sample rate
    model_mask += "sr{}".format(args.clients_sample_ratio)

    if args.homo_flag:
        model_mask = "homo_" + model_mask
        
    if args.enable_scheduler:
        model_mask += "_sf{}_step{}".format(args.scheduler_factor, args.scheduler_step)

    if args.softmax_flag:
        model_mask += "_softmaxLg"

    if args.public_flag:
        model_mask += "_public"

    if args.half_flag:
        model_mask += "_half"
        
    if args.resnet_flag:
        model_mask += "_rn"
    
    if args.old_CNN:
        model_mask += "_oldCNN"

    if args.VAE:
        model_mask += "_VAE{}".format(args.latent_size)
    
    if args.NVAE:
        model_mask += "_NoiseVAE"
        
    if args.new_VAE:
        model_mask += "_new_VAE"
        
    if args.VAE_op:
        model_mask += "_VAE_op"
        
    if args.new_VAE_ft:
        model_mask += "_VAE_ft"
        
    if args.GKT:
        model_mask += "_GKT"
    
    if args.VAE_fa:
        model_mask += "_VAE_fa"
        
    if args.op_agg:
        model_mask += "_op_agg"
    
    if args.op_agg_step:
        model_mask += "_op_agg_step"
        
    if args.MD:
        model_mask += "_MD"
        
    if args.op:
        model_mask += "_op"

    if args.second_highest:
        model_mask += "_SC"
    
    if args.Similarity:
        model_mask += "_Sim"
        
    
    
    # if args.Var:
    #     model_mask += "_Var{}".format(args.var_para)

    if args.middle_flag:
        model_mask += "_middle"

    if args.pre_train:
        model_mask += "_pt"

    if args.baseline_middle_flag:
        model_mask += "_baseline"

    if args.noniid_flag:
        model_mask += "_noniid"

    if args.KLD_flag:
        model_mask += "_KLD"
        
    
        

    # if not args.aggregate_flag:
    #     model_mask += "_noAgg"

    # if args.delayed_gradients_flag:
    #     model_mask += "_DG"
    # elif args.delayed_gradients_divided_flag:
    #     model_mask += "_DGDivided"
    # else:
    #     model_mask += "_notDealDG"

    if args.asyn_FL_flag:
        model_mask += "_asynFL"

    if args.y_distillation_flag and not args.no_logits_flag:
        model_mask += "_y_logits"

    if args.personalized_flag:
        model_mask += "_personalized"
    
    if args.distorted_data:
        model_mask += "_distorted"
    
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
            
        if args.enable_scheduler:
            based_model += "_schd"
            
        if args.homo_flag:
            cmp_log_dir = "./tb_logs/cmp_homo/" + based_model + "/" + model_mask + params_string
        else:
            cmp_log_dir = "./tb_logs/cmp/" + based_model + "/" + model_mask + params_string
        
    return cmp_log_dir
    
