import numpy as np
import os
# --------------------------------------------------------------------------------------------------
# SET parameters
df = 2.5  # fractal dimension
N = 1000  # number of beads
EPS = 1e-2  # nondimensional step time
INTERVAL = 10　# (EPS * INTERVAL) corresponds to the normalized time between adjacent frames
FRAME = 100  # frames of output XYZ file
SAMPLE = 10  # number of polymer dynamics samples
SEED = 12345678  # seed for the random number
# --------------------------------------------------------------------------------------------------
DIR = "data_dynamics"
os.makedirs(DIR, exist_ok=True)
NOISE = np.sqrt(2 * EPS)
F_Coefficient = 3 * EPS
np.random.seed(SEED)
# --------------------------------------------------------------------------------------------------


def Set_Q_lam_L():
    # Q
    Q = np.zeros((N, N))
    for i in range(N):
        for p in range(N):
            if p == 0:
                Q[i, p] = 1 / np.sqrt(N)
            else:
                Q[i, p] = np.sqrt(2 / N) * np.cos((i + 0.5) * p * np.pi / N)
    # 3*Σ^2
    threeSigma2 = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            if j == i:
                threeSigma2[i, j] = 0
            else:
                threeSigma2[i, j] = threeSigma2[j, i] = (j - i)**(2 / df)
    # M
    M = np.zeros((N, N))
    A_inv = (2 * N * np.eye(N, dtype=int) -
             np.ones((N, N), dtype=int)) / 2 / N**2
    M_diag = np.dot(np.dot(A_inv, threeSigma2), np.ones(N, dtype=int))
    for i in range(N):
        for j in range(i, N):
            M[i, j] = M[j, i] = (M_diag[i] + M_diag[j] - threeSigma2[i, j]) / 2
    # Λ
    Lam_inv = np.dot(Q.T, np.dot(M, Q))
    lam_inv = np.diag(Lam_inv)
    lam = np.zeros(N)
    lam[1:N] = 1 / lam_inv[1:N]
    Lam = np.diag(lam)
    # L
    L = np.dot(np.dot(Q, Lam), Q.T)
    return Q, lam, L
# --------------------------------------------------------------------------------------------------


def Equilibrium_Conformation_of_Normal_Coordinates(lam):
    Xx = np.zeros(N)
    Xy = np.zeros(N)
    Xz = np.zeros(N)
    for p in range(1, N):
        sd = np.sqrt(1 / 3 / lam[p])
        Xx[p] = sd * np.random.randn()
        Xy[p] = sd * np.random.randn()
        Xz[p] = sd * np.random.randn()
    return Xx, Xy, Xz
# --------------------------------------------------------------------------------------------------


def Convert_X_to_R(Xx, Xy, Xz, Q):
    Rx = np.dot(Q, Xx)
    Ry = np.dot(Q, Xy)
    Rz = np.dot(Q, Xz)
    return Rx, Ry, Rz
# --------------------------------------------------------------------------------------------------


def Integrate_Mode_Dynamics_without_CoM(x, y, z, lam):
    noise_x = NOISE * np.random.randn(N)
    noise_y = NOISE * np.random.randn(N)
    noise_z = NOISE * np.random.randn(N)

    force_x = - F_Coefficient * lam * x
    force_y = - F_Coefficient * lam * y
    force_z = - F_Coefficient * lam * z

    x_dt = x + force_x + noise_x
    y_dt = y + force_y + noise_y
    z_dt = z + force_z + noise_z

    force_x = - F_Coefficient * lam * x_dt
    force_y = - F_Coefficient * lam * y_dt
    force_z = - F_Coefficient * lam * z_dt

    x_2dt = x_dt + force_x + noise_x
    y_2dt = y_dt + force_y + noise_y
    z_2dt = z_dt + force_z + noise_z

    X = 0.5 * (x + x_2dt)
    Y = 0.5 * (y + y_2dt)
    Z = 0.5 * (z + z_2dt)

    X[0] = 0
    Y[0] = 0
    Z[0] = 0
    return X, Y, Z
# --------------------------------------------------------------------------------------------------


def Write_Psfdata():
    FILE_PSF = DIR + "/polymer_N{0:d}.psf".format(N)
    fp = open(FILE_PSF, "w")
    print("PSF\n\n       1 !NTITLE\n REMARKS\n", file=fp)
    print(" %7d !NATOM" % N, file=fp)
    for n in range(N):
        print(" %7d A    %04d GLY  CA   CT1    0.070000       12.0110           0"
              % (n + 1, n + 1), file=fp)
    print("\n %7d !NBOND: bonds" % (N - 1), file=fp)
    j = 0
    for i in range(N):
        if i % N != 0:
            print(" %7d %7d" % (i, i + 1), end="", file=fp)
            j += 1
            if j % 4 == 0:
                print("\n", end="", file=fp)
    print("\n", end="", file=fp)
    fp.close()
# --------------------------------------------------------------------------------------------------


def main():
    Write_Psfdata()
    Q, lam, L = Set_Q_lam_L()
    # ----------------------------------------------------------------------------------------------
    for sample in range(SAMPLE):
        Xx, Xy, Xz = Equilibrium_Conformation_of_Normal_Coordinates(lam)
        # ------------------------------------------------------------------------------------------
        FILE_OUT = DIR + "/df{0:.1f}_sample{1:d}.xyz".format(df, sample)
        fp = open(FILE_OUT, "w")
        for frame in range(FRAME + 1):
            Rx, Ry, Rz = Convert_X_to_R(Xx, Xy, Xz, Q)
            # --------------------------------------------------------------------------------------
            print("%d" % N, file=fp)
            print("frame = %d" % frame, file=fp)
            for n in range(N):
                print("CA\t%f\t%f\t%f" % (Rx[n], Ry[n], Rz[n]), file=fp)
            # --------------------------------------------------------------------------------------
            for step in range(INTERVAL):
                Xx, Xy, Xz = Integrate_Mode_Dynamics_without_CoM(Xx, Xy, Xz, lam)
        fp.close()
# --------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    main()
