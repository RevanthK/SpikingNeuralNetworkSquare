one_freq = .9
zero_freq = .05

def gen_st(num, arr_size, mult=1):
    
    if num:
        freq = one_freq
    else:
        freq = zero_freq

    if freq > .5:
        major = mult
        minor = 0
        freq = 1 - freq
    else:
        major = 0
        minor = mult

    spike_train = [major for x in range(arr_size)]

    num_minors = int(len(spike_train) * freq)
    num_majors = len(spike_train) - num_minors

    delta_majors = int(num_majors / num_minors)
    # print len(spike_train), num_majors, num_minors, delta_majors

    

    i = 0
    
    while i < len(spike_train):
        spike_train[i] = minor
        i += delta_majors
        num_minors -= 1

        i += 1

    return spike_train