import matplotlib.pyplot as plt
from hopfield_network import HopfieldNetwork


def main():
    n = 20
    hopfield_network = HopfieldNetwork(n)

    energy = [hopfield_network.energy]
    while not hopfield_network.is_local_minimum():
        hopfield_network.update()
        energy.append(hopfield_network.energy)

    plt.plot(energy)
    plt.show()


if __name__ == '__main__':
    main()
