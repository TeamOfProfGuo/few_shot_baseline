
def extract_feature(train_loader, val_loader, model, tag='last'):
    # return out mean, fcout mean, out feature, fcout features
    save_dir = '{}/{}/{}'.format(args.save_path, tag, args.enlarge)
    if os.path.isfile(save_dir + '/output.plk'):
        data = load_pickle(save_dir + '/output.plk')
        return data
    else:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

    model.eval()
    with torch.no_grad():
        # get training mean
        out_mean, fc_out_mean = [], []
        for i, (inputs, _) in enumerate(warp_tqdm(train_loader)):
            outputs, fc_outputs = model(inputs, True)
            out_mean.append(outputs.cpu().data.numpy())
            if fc_outputs is not None:
                fc_out_mean.append(fc_outputs.cpu().data.numpy())
        out_mean = np.concatenate(out_mean, axis=0).mean(0)
        if len(fc_out_mean) > 0:
            fc_out_mean = np.concatenate(fc_out_mean, axis=0).mean(0)
        else:
            fc_out_mean = -1

        output_dict = collections.defaultdict(list)
        fc_output_dict = collections.defaultdict(list)
        for i, (inputs, labels) in enumerate(warp_tqdm(val_loader)):
            # compute output
            outputs, fc_outputs = model(inputs, True)
            outputs = outputs.cpu().data.numpy()
            if fc_outputs is not None:
                fc_outputs = fc_outputs.cpu().data.numpy()
            else:
                fc_outputs = [None] * outputs.shape[0]
            for out, fc_out, label in zip(outputs, fc_outputs, labels):
                output_dict[label.item()].append(out)
                fc_output_dict[label.item()].append(fc_out)
        all_info = [out_mean, fc_out_mean, output_dict, fc_output_dict]
        save_pickle(save_dir + '/output.plk', all_info)
        return all_info