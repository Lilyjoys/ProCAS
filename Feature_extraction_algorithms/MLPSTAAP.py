
import numpy as np
import scipy.io
import os


def PSTAAP_feature(protein_sequences, test_PSTAAP=False):
    for i in range(len(protein_sequences)):
        protein_sequences[i] = protein_sequences[i][:24] + protein_sequences[i][25:]

    if test_PSTAAP:
        mat_contents = scipy.io.loadmat("Feature_extraction_algorithms/Fr_test.mat")
    else:
        mat_contents = scipy.io.loadmat("Feature_extraction_algorithms/Fr_train.mat")

    Fr = mat_contents['Fr']
    """
    print(Fr[0*400+5*20+0,0])
    print(Fr[5 * 400 + 0 * 20 + 16, 1])
    print(Fr[0 * 400 + 16 * 20 + 14, 2])
    """
    AA = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    PSTAAP = np.zeros((len(protein_sequences), 46))
    for i in range(len(protein_sequences)):
        for j in range(len(protein_sequences[0])-2):
            t1 = protein_sequences[i][j]
            position1 = AA.index(t1)
            t2 = protein_sequences[i][j+1]
            position2 = AA.index(t2)
            t3 = protein_sequences[i][j+2]
            position3 = AA.index(t3)

            PSTAAP[i][j] = Fr[400 * position1 + 20 * position2 + position3][j]

    return PSTAAP


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import splrep, BSpline
    from sklearn.preprocessing import MinMaxScaler
    from numpy.polynomial import Polynomial


    def plot_multiple_polynomial_fitted_functions(sample_datas, degree=3):
        markers = ["o", "o", "^", "^", "v", "p"]
        colors = ["b", "b", "c", "c", "m", "y"]
        label = ["sample1(1,0,0,0)", "sample1(1,0,0,0)", "sample1(0,1,0,0)", "sample2(0,1,0,0)", "sample3(0,0,1,0)", "sample6(0,0,0,1)"]
        plt.figure(figsize=(12, 6))

        for i, sample_data in enumerate(sample_datas):
            if i == 0 or i == 1 or i == 4 or i == 5:
                continue

            scaler = MinMaxScaler()
            normalized_data = scaler.fit_transform(sample_data.reshape(-1, 1)).flatten()

            x = np.linspace(0, 1, len(normalized_data))
            p = Polynomial.fit(x, normalized_data, degree)
            y_poly = p(x)

            dy_poly = p.deriv(1)(x)
            extrema_indices = np.where(np.diff(np.sign(dy_poly)))[0]
            extrema_x = x[extrema_indices]
            extrema_y = y_poly[extrema_indices]

            plt.plot(x, y_poly, label=f'{label[i]}', marker=markers[i], color=colors[i])
            plt.plot(extrema_x, extrema_y, 'rx', markersize=10)

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Fitted Polynomial Functions with Extrema')
        plt.legend()
        plt.show()

    def plot_multiple_fitted_functions(sample_datas, smooth_factor=1):
        markers = ["o", "o", "^", "^", "v", "p"]
        colors = ["b", "b", "c", "c", "m", "y"]
        label = ["", "", "sample1(0,1,0,0)", "sample2(0,1,0,0)", "sample3(0,0,1,0)", ""]
        plt.figure(figsize=(12, 6))

        for i, sample_data in enumerate(sample_datas):
            if i == 0 or i == 1 or i == 4 or i == 5:
                continue

            scaler = MinMaxScaler()
            normalized_data = scaler.fit_transform(sample_data.reshape(-1, 1)).flatten()

            x = np.linspace(0, 1, len(normalized_data))
            tck = splrep(x, normalized_data, k=3, s=smooth_factor)
            spline = BSpline(tck[0], tck[1], tck[2])

            y_spline = spline(x)
            dy_spline = spline.derivative()
            extrema_indices = np.where(np.diff(np.sign(dy_spline(x))))[0]
            extrema_x = x[extrema_indices]
            extrema_y = y_spline[extrema_indices]

            plt.plot(x, y_spline, label=f'{label[i]}', marker=markers[i], color=colors[i])
            plt.plot(extrema_x, extrema_y, 'rx', markersize=10)

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Fitted B-Spline Functions with Extrema')
        plt.legend()
        plt.show()

    protein_sequences = [
        "TSPASVASSSSTPSSKTKDLGHNDKSSTPGLKSNTPTPRNDAPTPGTST",  # a
        "LGGNIEQLVARSNILTLMYQCMQDKMPEVRQSSFALLGDLTKACFQHVK",  # a
        "VDFQHASEDARKTINQWVKGQTEGKIPELLASGMVDNMTKLVLVNAIYF",  # c
        "VEGTLKGPEVDLKGPRLDFEGPDAKLSGPSLKMPSLEISAPKVTAPDVD",  # c
        "IDILTSREQFFSDEERKYMAINQKKAYILVTPLKSRKVIEQRCMRYNLS",  # m
        "LAGTDGETTTQGLDGLSERCAQYKKDGADFAKWRCVLKISERTPSALAI",  # s
    ]
    data = PSTAAP_feature(protein_sequences, False)


    plot_multiple_polynomial_fitted_functions(data)
    plot_multiple_fitted_functions(data)
