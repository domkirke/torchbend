
def linear(module, interp_dict):
    state_dict = None
    weight_sum = sum([interp_dict[k][1] for k in interp_dict.keys()])
    for c, (d, w) in interp_dict.items():
        if state_dict is None:
            state_dict = {k: v  * w / weight_sum for k, v in d.items()}
        else:
            for k, v in d.items():
                state_dict[k] = state_dict[k] + v * w / weight_sum
    return state_dict



