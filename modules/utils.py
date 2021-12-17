



def get_model_mask(args):

    if args.GAN_moment_flag:
        model_mask = "moment"
    else:
        model_mask = "Adam"

    # sample rate
    model_mask += "sr{}".format(args.clients_sample_ratio)

    if args.homo_flag:
        model_mask = "homo_" + model_mask

    if args.softmax_flag:
        model_mask += "_softmaxLg"

    if args.public_flag:
        model_mask += "_public"

    if args.BN_features:
        model_mask += "_BN"

    if args.half_flag:
        model_mask += "_half"
        
    if args.resnet_flag:
        model_mask += "_rn"

    if args.VAE:
        model_mask += "_VAE{}".format(args.latent_size)

    if args.second_highest:
        model_mask += "_SC"
    
    if args.Similarity:
        model_mask += "_Sim"
    
    if args.Var:
        model_mask += "_Var{}".format(args.var_para)

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

    if args.y_distillation_flag:
        model_mask += "_y_logits"

    if args.personalized_flag:
        model_mask += "_personalized"
    
    if args.distorted_data:
        model_mask += "_distorted"
    
    return model_mask
