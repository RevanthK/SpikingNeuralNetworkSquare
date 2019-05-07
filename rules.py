hebb = .01
sig = .0000001
w_max = 1
w_min = 0


def ojas(v_i, v_j, w_ij):
    return hebb * (v_i * v_j - w_ij * (v_j ** 2))

def stdp(w, del_t):
    if del_t < 0:
        return w + sig * del_t * (w - w_min)
    elif del_t > 0:
        return w + sig * del_t * (w_max - w)
    else:
        return w