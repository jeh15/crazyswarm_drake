import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pdb

def main() -> None:
    # resolution = 11
    # x = np.linspace(0.01, 0.25, resolution)
    # scaler = 10
    # offset = -np.log(1)
    # func = np.exp(-(scaler * x + offset))
    # log_func = np.log(1-func)

    resolution = 2
    x = np.linspace(0.001, 0.25, resolution)
    func = -x + 1
    log_func = np.log(1-func)
    p_list = []
    for i in range(0, resolution-1):
        p = np.polyfit(x[i:i+2], log_func[i:i+2], 1)
        p_list.append(p)

    p_array = np.asarray(p_list)
    m_ = p_array[:, 0]
    b_ = p_array[:, 1]

    sv = np.linspace(0, 1, 21)
    xv = np.linspace(0, 1, 21)

    m_test = np.ones((2,))
    b_test = np.array([1, 2])
    m_x = np.einsum('i,j->ij', m_test, xv)
    m_x_b = m_x + b_test.reshape(2, -1)
    m_x_b_sv = sv - m_x_b
    
    res = (
        -((np.einsum('i,j->ij', m_test, xv)) + b_test.reshape(b_test.shape[0], -1)) + sv
        ).flatten()


    ic = np.array([0, 1, 2, 3, 4, 5])
    proj = (np.einsum('i,j->ij', ic[3:5], sv)) + ic[:2].reshape((2, -1))

    prev_traj = np.array(
        [
            np.ones((21,)),
            2*np.ones((21,)),
        ],
    )

    n_ = proj - prev_traj

    res = np.einsum('ij,ij->j', n_, n_)
    
    pdb.set_trace()

    plt.figure()
    plt.plot(x, log_func)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


if __name__ == "__main__":
    main()