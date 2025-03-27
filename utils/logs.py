from utils.metrics import similarity

def check_list(e_features, d_features, model_path, 
               sparse_e1_run, sparse_e2_run, sparse_e3_run, sparse_e4_run, sparse_e5_run,
               sparse_d1_run, sparse_d2_run, sparse_d3_run, sparse_d4_run, sparse_d5_run,
               sim1_run, sim2_run, sim3_run, sim4_run, sim5_run):

    result = similarity(e_features, d_features, model_path)
    total_sim, total_sparse_e, total_sparse_d =  result['sim'], result['spars_e'], result['spars_d']
    sim1, sim2, sim3, sim4, sim5 = total_sim[-1], total_sim[-2], total_sim[-3], total_sim[-4], total_sim[-5]
    spars_e1, spars_e2, spars_e3, spars_e4, spars_e5 = total_sparse_e[-1], total_sparse_e[-2], total_sparse_e[-3], total_sparse_e[-4], total_sparse_e[-5]
    spars_d1, spars_d2, spars_d3, spars_d4, spars_d5 = total_sparse_d[-1], total_sparse_d[-2], total_sparse_d[-3], total_sparse_d[-4], total_sparse_d[-5]

    sparse_e1_run.update(spars_e1)
    sparse_e2_run.update(spars_e2)
    sparse_e3_run.update(spars_e3)
    sparse_e4_run.update(spars_e4)
    sparse_e5_run.update(spars_e5)

    sparse_d1_run.update(spars_d1)
    sparse_d2_run.update(spars_d2)
    sparse_d3_run.update(spars_d3)
    sparse_d4_run.update(spars_d4)
    sparse_d5_run.update(spars_d5)

    sim1_run.update(sim1)
    sim2_run.update(sim2)
    sim3_run.update(sim3)
    sim4_run.update(sim4)
    sim5_run.update(sim5)

    return sim1, sim2


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
