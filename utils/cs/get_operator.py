import torch


class H:
    def __init__(self, rand_pm_vec_list, perm_idx_list, rep_level=4):
        self.rand_pm_vec_list = rand_pm_vec_list
        self.perm_idx_list = perm_idx_list
        self.rep_level = rep_level

    def A(self, X):
        X_i = X.reshape(X.shape[0], -1)
        for j in range(self.rep_level):
            X_foo_1 = X_i[:, :, self.perm_idx_list[0]]
            X_foo_2 = X_foo_1 * self.rand_pm_vec_list[0]
            X_i = (torch.view_as_real(
                torch.fft.fft(torch.view_as_complex(X_foo_2.permute(1, 0).contiguous()), norm="ortho"))).permute(1, 0).contiguous()
        return X_i.reshape(X.shape)

    def H(self, Y):
        Y_i = Y.reshape(Y.shape[0], -1)
        for j in range(self.rep_level):
            Y_foo_1 = (torch.view_as_real(
                torch.fft.ifft(torch.view_as_complex(Y_i.permute(1, 0).contiguous()), norm="ortho"))).permute(1, 0).contiguous()
            Y_foo_2 = Y_foo_1 * self.rand_pm_vec_list[0]
            Y_foo_3 = torch.zeros_like(Y_foo_2)
            Y_foo_3[:, :, self.perm_idx_list[0]] = Y_foo_2
            Y_i = Y_foo_3
        return Y_i.reshape(Y.shape)


def get_operator(N, mask_type=2):
    if mask_type == 1:
        return None
    elif mask_type == 2:
        rand_pm_vec_list = []
        pm_vec = (2 * (torch.rand(N) < 0.5) - 1)
        rand_pm_vec_list.append(pm_vec.repeat(2, 1))

        perm_idx_list = []
        perm_idx_list.append(torch.randperm(N))

        return H(rand_pm_vec_list, perm_idx_list)
