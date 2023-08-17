import contextlib
import types
import warnings

import numpy as np
import torch

from neuralpredictors.measures import corr
import neuralpredictors.measures as losses
from neuralpredictors.training import device_state, eval_state


def model_predictions_repeats(model, dataloader, data_key, device="cpu", broadcast_to_target=False):
    """
    Computes model predictions for a dataloader that yields batches with identical inputs along the first dimension.
    Unique inputs will be forwarded only once through the model
    Returns:
        target: ground truth, i.e. neuronal firing rates of the neurons as a list: [num_images][num_reaps, num_neurons]
        output: responses as predicted by the network for the unique images. If broadcast_to_target, returns repeated
                outputs of shape [num_images][num_reaps, num_neurons] else (default) returns unique outputs of shape [num_images, num_neurons]
    """

    target = []
    unique_images = torch.empty(0)
    for datapoint in dataloader:
        images = datapoint.images.to(device) if "images" in datapoint._fields else datapoint.inputs.to(device)
        responses = datapoint.responses.to(device) if "responses" in datapoint._fields else datapoint.targets.to(device)

        if len(images.shape) == 5:
            images = images.squeeze(dim=0)
            responses = responses.squeeze(dim=0)

        assert torch.all(
            torch.eq(
                images[
                    -1,
                ],
                images[
                    0,
                ],
            )
        ), "All images in the batch should be equal"
        unique_images = torch.cat(
            (
                unique_images,
                images[
                    0:1,
                ]
                .detach()
                .cpu(),
            ),
            dim=0,
        )
        target.append(responses.detach().cpu().numpy())

    # Forward unique images once:
    with eval_state(model) if not isinstance(model, types.FunctionType) else contextlib.nullcontext():
        with device_state(model, device) if not isinstance(model, types.FunctionType) else contextlib.nullcontext():
            output = model.predict_mean(unique_images.to(device), data_key=data_key).detach().cpu()

    output = output.numpy()

    if broadcast_to_target:
        output = [np.broadcast_to(x, target[idx].shape) for idx, x in enumerate(output)]

    return target, output


def model_predictions(model, dataloader, data_key, device="cpu"):
    """
    computes model predictions for a given dataloader and a model
    Returns:
        target: ground truth, i.e. neuronal firing rates of the neurons
        output: responses as predicted by the network
    """

    with torch.no_grad():
        with device_state(model, device) if not isinstance(model, types.FunctionType) else contextlib.nullcontext():
            outputs, targets = [], []
            for b in dataloader:
                pupil_center = b.pupil_center.to(device) if hasattr(b, "pupil_center") else None
                behavior = b.behavior.to(device) if hasattr(b, "behavior") else None
                images = b.images.to(device) if "images" in b._fields else b.inputs.to(device)
                responses = b.responses if "responses" in b._fields else b.targets.to(device)
                output = model.predict_mean(images, data_key=data_key, pupil_center=pupil_center, behavior=behavior).cpu().data.numpy()
                output = np.repeat(output, responses.shape[0], axis=0) if output.shape[0] == 1 else output
                outputs.append(output)
                targets.append(
                    model.transform(responses, data_key=data_key)[0].cpu().data.numpy()
                    if hasattr(model, "transform")
                    else responses.cpu().data.numpy()
                )
            outputs = np.vstack(outputs)
            targets = np.vstack(targets)

    return targets, outputs


def get_avg_correlations(model, dataloaders, device="cpu", as_dict=False, per_neuron=True, **kwargs):
    """
    Returns correlation between model outputs and average responses over repeated trials

    """
    if "test" in dataloaders:
        dataloaders = dataloaders["test"]

    correlations = {}
    for k, loader in dataloaders.items():

        # Compute correlation with average targets
        target, output = model_predictions_repeats(
            dataloader=loader,
            model=model,
            data_key=k,
            device=device,
            broadcast_to_target=False,
        )
        target_mean = np.array([t.mean(axis=0) for t in target])
        correlations[k] = corr(target_mean, output, axis=0)

        # Check for nans
        if np.any(np.isnan(correlations[k])):
            warnings.warn("{}% NaNs , NaNs will be set to Zero.".format(np.isnan(correlations[k]).mean() * 100))
        correlations[k][np.isnan(correlations[k])] = 0

    if not as_dict:
        correlations = (
            np.hstack([v for v in correlations.values()])
            if per_neuron
            else np.mean(np.hstack([v for v in correlations.values()]))
        )
    return correlations


def get_correlations(model, dataloaders, device="cpu", as_dict=False, per_neuron=True, **kwargs):
    correlations = {}
    with eval_state(model) if not isinstance(model, types.FunctionType) else contextlib.nullcontext():
        for k, v in dataloaders.items():

            target, output = model_predictions(dataloader=v, model=model, data_key=k, device=device)
            correlations[k] = corr(target, output, axis=0)

            if np.any(np.isnan(correlations[k])):
                warnings.warn("{}% NaNs , NaNs will be set to Zero.".format(np.isnan(correlations[k]).mean() * 100))
            correlations[k][np.isnan(correlations[k])] = 0

    if not as_dict:
        correlations = (
            np.hstack([v for v in correlations.values()])
            if per_neuron
            else np.mean(np.hstack([v for v in correlations.values()]))
        )
    return correlations


def get_loss(
    model,
    dataloaders,
    loss_function,
    device="cpu",
    as_dict=False,
    avg=False,
    per_neuron=True,
):
    loss_config = dict(avg=avg, per_neuron=per_neuron)
    if loss_function == "PoissonLoss":
        loss_config.update({"full_loss": True})
    loss_fn = getattr(losses, loss_function)(**loss_config)

    loss_vals = {}
    with eval_state(model) if not isinstance(model, types.FunctionType) else contextlib.nullcontext():
        for k, v in dataloaders.items():
            loss = []
            n_trials = 0
            for b in v:
                images = b.images.to(device) if "images" in b._fields else b.inputs.to(device)
                responses = b.responses.to(device) if "responses" in b._fields else b.targets.to(device)
                pupil_center = b.pupil_center.to(device) if hasattr(b, "pupil_center") else None
                behavior = b.behavior.to(device) if hasattr(b, "behavior") else None
                n_trials += images.shape[0]
                if hasattr(model, "transform"):
                    loss.append(
                        loss_fn(
                            target=model.transform(responses.to(device), data_key=k),
                            output=model(images.to(device), data_key=k, pupil_center=pupil_center, behavior=behavior),
                        )
                        .cpu()
                        .data.numpy()
                    )
                else:
                    loss.append(
                        loss_fn(
                            target=responses.to(device),
                            output=model(images.to(device), data_key=k, pupil_center=pupil_center, behavior=behavior),
                        )
                        .cpu()
                        .data.numpy()
                    )

            loss = np.vstack(loss)
            loss = np.mean(loss, axis=0) if avg else np.sum(loss, axis=0)
            loss_vals[k] = loss

    if as_dict:
        return loss_vals
    else:
        loss_per_neuron = np.hstack([v for v in loss_vals.values()])
        if per_neuron:
            return loss_per_neuron
        else:
            return (
                np.mean(loss_per_neuron)
                if avg
                else np.sum(loss_per_neuron))



def get_model_performance(model, dataloaders, loss_function, device="cpu", print_performance=True):
    output = {"correlation": {}, "loss": {}}
    for tier in ["train", "validation", "test"]:
        output["correlation"][tier] = get_correlations(model, dataloaders[tier], device=device, per_neuron=False)

        output["loss"][tier] = get_loss(
            model,
            dataloaders[tier],
            loss_function,
            device=device,
            per_neuron=False,
            avg=True,
        )
    if print_performance:
        for measure, tiers in output.items():
            print("\u0332".join(measure + " "))
            print("")
            for tier, value in tiers.items():
                print(tier + ":" + " " * (13 - len(tier)) + "{0:.3f} ".format(value))
            print("")


def get_repeats(dataloader, min_repeats=2):
    # save the responses of all neuron to the repeats of an image as an element in a list
    repeated_inputs = []
    repeated_outputs = []
    for inputs, outputs in dataloader:
        if len(inputs.shape) == 5:
            inputs = np.squeeze(inputs.cpu().numpy(), axis=0)
            outputs = np.squeeze(outputs.cpu().numpy(), axis=0)
        else:
            inputs = inputs.cpu().numpy()
            outputs = outputs.cpu().numpy()
        r, n = outputs.shape  # number of frame repeats, number of neurons
        if r < min_repeats:  # minimum number of frame repeats to be considered for oracle, free choice
            continue
        assert np.all(np.abs(np.diff(inputs, axis=0)) == 0), "Images of oracle trials do not match"
        repeated_inputs.append(inputs)
        repeated_outputs.append(outputs)
    return np.array(repeated_inputs), np.array(repeated_outputs)


def get_oracles(dataloaders, as_dict=False, per_neuron=True):
    oracles = {}
    for k, v in dataloaders.items():
        _, outputs = get_repeats(v)
        oracles[k] = compute_oracle_corr(np.array(outputs))
    if not as_dict:
        oracles = (
            np.hstack([v for v in oracles.values()])
            if per_neuron
            else np.mean(np.hstack([v for v in oracles.values()]))
        )
    return oracles


def get_oracles_corrected(dataloaders, as_dict=False, per_neuron=True):
    oracles = {}
    for k, v in dataloaders.items():
        _, outputs = get_repeats(v)
        oracles[k] = compute_oracle_corr_corrected(np.array(outputs))
    if not as_dict:
        oracles = (
            np.hstack([v for v in oracles.values()])
            if per_neuron
            else np.mean(np.hstack([v for v in oracles.values()]))
        )
    return oracles


def compute_oracle_corr_corrected(repeated_outputs):
    if len(repeated_outputs.shape) == 3:
        var_noise = repeated_outputs.var(axis=1).mean(0)
        var_mean = repeated_outputs.mean(axis=1).var(0)
    else:
        var_noise, var_mean = [], []
        for repeat in repeated_outputs:
            var_noise.append(repeat.var(axis=0))
            var_mean.append(repeat.mean(axis=0))
        var_noise = np.mean(np.array(var_noise), axis=0)
        var_mean = np.var(np.array(var_mean), axis=0)
    return var_mean / np.sqrt(var_mean * (var_mean + var_noise))


def compute_oracle_corr(repeated_outputs):
    if len(repeated_outputs.shape) == 3:
        _, r, n = repeated_outputs.shape
        oracles = (repeated_outputs.mean(axis=1, keepdims=True) - repeated_outputs / r) * r / (r - 1)
        if np.any(np.isnan(oracles)):
            warnings.warn(
                "{}% NaNs when calculating the oracle. NaNs will be set to Zero.".format(np.isnan(oracles).mean() * 100)
            )
        oracles[np.isnan(oracles)] = 0
        return corr(oracles.reshape(-1, n), repeated_outputs.reshape(-1, n), axis=0)
    else:
        oracles = []
        for outputs in repeated_outputs:
            r, n = outputs.shape
            # compute the mean over repeats, for each neuron
            mu = outputs.mean(axis=0, keepdims=True)
            # compute oracle predictor
            oracle = (mu - outputs / r) * r / (r - 1)

            if np.any(np.isnan(oracle)):
                warnings.warn(
                    "{}% NaNs when calculating the oracle. NaNs will be set to Zero.".format(
                        np.isnan(oracle).mean() * 100
                    )
                )
                oracle[np.isnan(oracle)] = 0

            oracles.append(oracle)
        return corr(np.vstack(repeated_outputs), np.vstack(oracles), axis=0)


def get_fraction_oracles(model, dataloaders, device="cpu", corrected=False):
    dataloaders = dataloaders["test"] if "test" in dataloaders else dataloaders
    if corrected:
        oracles = get_oracles_corrected(dataloaders=dataloaders, as_dict=False, per_neuron=True)
    else:
        oracles = get_oracles(dataloaders=dataloaders, as_dict=False, per_neuron=True)
    test_correlation = get_correlations(
        model=model, dataloaders=dataloaders, device=device, as_dict=False, per_neuron=True
    )
    oracle_performance, _, _, _ = np.linalg.lstsq(np.hstack(oracles)[:, np.newaxis], np.hstack(test_correlation))
    return oracle_performance


def get_r2er(model, dataloaders, device="cpu"):
    dataloaders = dataloaders["test"] if "test" in dataloaders else dataloaders
    r2er = {}
    for data_key, dataloader in dataloaders.items():
        # get targets and predictions
        target, output = model_predictions_repeats(model, dataloader, data_key, device=device)
        target = fill_response_repeats(target)
        # re-arrange arrays so they fit for the function r2er_n2m
        target = np.moveaxis(target, [0, 1], [-1, -2])
        output = np.moveaxis(output, [0, 1], [-1, -2])
        # compute r2er
        r2er[data_key], _ = compute_r2er_n2m(output, target)

    # TODO: This has to be adapted to allow for per_neuron, as_dict, etc...
    r2er = np.mean(np.hstack([v for v in r2er.values()]))
    return r2er


def get_feve(model, dataloaders, device="cpu"):
    dataloaders = dataloaders["test"] if "test" in dataloaders else dataloaders
    feve = {}
    for data_key, dataloader in dataloaders.items():
        # get targets and predictions
        target, output = model_predictions_repeats(model, dataloader, data_key, device=device)
        target = fill_response_repeats(target)
        output = np.repeat(output[:, np.newaxis, :], target.shape[1], axis=1)
        # compute feve
        feve[data_key] = compute_feve(target, output)

    # TODO: This has to be adapted to allow for per_neuron, as_dict, etc...
    feve = np.mean(np.hstack([v for v in feve.values()]))
    return feve


def get_explainable_var(dataloaders, as_dict=False, per_neuron=True):
    dataloaders = dataloaders["test"] if "test" in dataloaders else dataloaders
    explainable_var = {}
    for k, v in dataloaders.items():
        _, outputs = get_repeats(v)
        explainable_var[k] = compute_explainable_var(outputs)
    if not as_dict:
        explainable_var = (
            np.hstack([v for v in explainable_var.values()])
            if per_neuron
            else np.mean(np.hstack([v for v in explainable_var.values()]))
        )
    return explainable_var


def compute_explainable_var(outputs, eps=1e-9):
    ImgVariance = []
    TotalVar = np.var(np.vstack(outputs), axis=0, ddof=1)
    for out in outputs:
        ImgVariance.append(np.var(out, axis=0, ddof=1))
    ImgVariance = np.vstack(ImgVariance)
    NoiseVar = np.mean(ImgVariance, axis=0)
    explainable_var = (TotalVar - NoiseVar) / (TotalVar + eps)
    return explainable_var


def get_FEV(model, dataloaders, device="cpu", as_dict=False, per_neuron=True, threshold=None):
    """
    Computes the fraction of explainable variance explained (FEVe) per Neuron, given a model and a dictionary of dataloaders.
    The dataloaders will have to return batches of identical images, with the corresponing neuronal responses.
    Args:
        model (object): PyTorch module
        dataloaders (dict): Dictionary of dataloaders, with keys corresponding to "data_keys" in the model
        device (str): 'cuda' or 'gpu
        as_dict (bool): Returns the scores as a dictionary ('data_keys': values) if set to True.
        per_neuron (bool): Returns the grand average if set to True.
        threshold (float): for the avg feve, excludes neurons with a explainable variance below threshold
    Returns:
        FEV (dict, or np.array, or float): Fraction of explainable varianced explained. Per Neuron or as grand average.
    """
    dataloaders = dataloaders["test"] if "test" in dataloaders else dataloaders
    FEV = {}
    with eval_state(model) if not isinstance(model, types.FunctionType) else contextlib.nullcontext():
        for data_key, dataloader in dataloaders.items():
            targets, outputs = model_predictions_repeats(
                model=model, dataloader=dataloader, data_key=data_key, device=device, broadcast_to_target=True
            )
            if threshold is None:
                FEV[data_key] = compute_FEV(targets=targets, outputs=outputs)
            else:
                fev, feve = compute_FEV(targets=targets, outputs=outputs, return_exp_var=True)
                FEV[data_key] = feve[fev > threshold]
    if not as_dict:
        FEV = np.hstack([v for v in FEV.values()]) if per_neuron else np.mean(np.hstack([v for v in FEV.values()]))
    return FEV


def compute_FEV(targets, outputs, return_exp_var=False):
    """
    Args:
        targets (list): Neuronal responses (ground truth) to image repeats. Dimensions: [num_images] np.array(num_reaps, num_neurons)
        outputs (list): Model predictions to the repeated images, with an identical shape as the targets
        return_exp_var (bool): returns the fraction of explainable variance per neuron if set to True
    Returns:
        FEVe (np.array): the fraction of explainable variance explained per neuron
        --- optional: FEV (np.array): the fraction
    """
    ImgVariance = []
    PredVariance = []
    for i, _ in enumerate(targets):
        PredVariance.append((targets[i] - outputs[i]) ** 2)
        ImgVariance.append(np.var(targets[i], axis=0, ddof=1))
    PredVariance = np.vstack(PredVariance)
    ImgVariance = np.vstack(ImgVariance)

    TotalVar = np.var(np.vstack(targets), axis=0, ddof=1)
    NoiseVar = np.mean(ImgVariance, axis=0)
    FEV = (TotalVar - NoiseVar) / TotalVar

    PredVar = np.mean(PredVariance, axis=0)
    FEVe = 1 - (PredVar - NoiseVar) / (TotalVar - NoiseVar)
    return [FEV, FEVe] if return_exp_var else FEVe


def get_cross_oracles(data, reference_data):
    _, outputs = get_repeats(data)
    _, outputs_reference = get_repeats(reference_data)
    cross_oracles = compute_cross_oracles(outputs, outputs_reference)
    return cross_oracles


def compute_cross_oracles(repeats, reference_data):
    pass


def normalize_RGB_channelwise(mei):
    mei_copy = mei.copy()
    mei_copy = mei_copy - mei_copy.min(axis=(1, 2), keepdims=True)
    mei_copy = mei_copy / mei_copy.max(axis=(1, 2), keepdims=True)
    return mei_copy


def normalize_RGB(mei):
    mei_copy = mei.copy()
    mei_copy = mei_copy - mei_copy.min()
    mei_copy = mei_copy / mei_copy.max()
    return mei_copy


def get_model_rf_size(model_config):
    layers = model_config["layers"]
    input_kern = model_config["input_kern"]
    hidden_kern = model_config["hidden_kern"]
    dil = model_config["hidden_dilation"]
    rf_size = input_kern + ((hidden_kern - 1) * dil) * (layers - 1)
    return rf_size


def get_predictions(model, dataloaders, device="cpu", as_dict=False, per_neuron=True, test_data=False, **kwargs):
    predictions = {}
    with eval_state(model) if not isinstance(model, types.FunctionType) else contextlib.nullcontext():
        for k, v in dataloaders.items():
            if test_data:
                _, output = model_predictions_repeats(dataloader=v, model=model, data_key=k, device=device)
            else:
                _, output = model_predictions(dataloader=v, model=model, data_key=k, device=device)
            predictions[k] = output.T

    if not as_dict:
        predictions = [v for v in predictions.values()]
    return predictions


def get_targets(model, dataloaders, device="cpu", as_dict=True, per_neuron=True, test_data=False, **kwargs):
    responses = {}
    with eval_state(model) if not isinstance(model, types.FunctionType) else contextlib.nullcontext():
        for k, v in dataloaders.items():
            if test_data:
                targets, _ = model_predictions_repeats(dataloader=v, model=model, data_key=k, device=device)
                targets_per_neuron = []
                for i in range(targets[0].shape[1]):
                    neuronal_responses = []
                    for repeats in targets:
                        neuronal_responses.append(repeats[:, i])
                    targets_per_neuron.append(neuronal_responses)
                responses[k] = targets_per_neuron
            else:
                targets, _ = model_predictions(dataloader=v, model=model, data_key=k, device=device)
                responses[k] = targets.T

    if not as_dict:
        responses = [v for v in responses.values()]
    return responses


def fill_response_repeats(x, fillval=np.nan):
    """
    Takes in the responses as provided by 'model_predictions_repeats' and fills the missing repeats per unique image with the fillval.
    """
    lens = np.array([len(item) for item in x])
    shape = np.array(x)[lens == lens.max()][0].shape
    for idx in np.where(lens != lens.max())[0]:
        helper_array = np.full(shape, fillval)
        helper_array[: lens[idx], :] = x[idx]
        x[idx] = helper_array
    return np.stack(x)


def compute_r2er_n2m(x, y):
    """
    Approximately unbiased estimator of r^2 between the expected values.
        of the rows of x and y. Assumes x is fixed and y has equal variance across
        trials and observations
    Parameters
    ----------
    x : numpy.ndarray
        m unique images model predictions
    y : numpy.ndarray
        n repeats by m unique images array of data

    Returns
    -------
    r2er : an estimate of the r2 between the expected values
    r2 :   classic r2
    --------
    """
    n, m = np.shape(y)[-2:]
    # estimate trial to trial variability for each stim then average across all
    sigma2 = np.nanmean(np.nanvar(y, -2, ddof=1, keepdims=True), -1)

    # center predictions
    x_ms = x - np.nanmean(x, -1, keepdims=True)
    # get average responses across trials
    y = np.nanmean(y, -2, keepdims=True)
    # center responses
    y_ms = np.squeeze(y - np.nanmean(y, -1, keepdims=True))
    # get sample covariance squared between prediction and responses
    xy2 = np.nansum((x_ms * y_ms), -1, keepdims=True) ** 2
    # get variance for model and responses
    x2 = np.nansum(x_ms**2, -1, keepdims=True)
    y2 = np.nansum(y_ms**2, -1, keepdims=True)
    x2y2 = x2 * y2

    # classic r2
    r2 = xy2 / x2y2

    # subtract off estimates of bias for numerator and denominator
    ub_xy2 = xy2 - sigma2 / n * x2
    ub_x2y2 = x2y2 - (m - 1) * sigma2 / n * x2

    # form ratio of unbiased estimates
    r2er = ub_xy2 / ub_x2y2

    return np.squeeze(r2er), r2


def compute_feve(target, output):
    """
    Computes the fraction of explainable variance explained as done in
    Cadena et al 2019 and Burg et al 2020.
    Args:
        target: (images, repeats, neurons)    Observed neural responses
        output: (images, repeats, neurons)  Model predictions
    Returns:
        feve: (neurons) Fraction of explainable variance explained
    """

    sqr_err, obs_var = [], []
    for t, o in zip(target, output):  # loop over the images
        sqr_err.append((t - o) ** 2)  # squared error per image
        obs_var.append(np.nanvar(t, axis=0, ddof=1))  # conditional variance across repetitions

    mse = np.nanmean(np.array(sqr_err), axis=(0, 1))  # mean squared error
    mean_obs_var = np.mean(np.array(obs_var), axis=0)  # mean obs variance

    total_var = np.nanvar(target, axis=(0, 1), ddof=1)  # total variance
    explainable_var = total_var - mean_obs_var  # explainable variance

    feve = 1 - (mse - mean_obs_var) / explainable_var  # fraction of explainable variance explained

    return feve
