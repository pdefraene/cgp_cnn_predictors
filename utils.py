import matplotlib.pyplot as plt

label_names = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]


def plot_images(images, cls_true, cls_pred=None):
    """
    Adapted from https://github.com/Hvass-Labs/TensorFlow-Tutorials/
    """
    fig, axes = plt.subplots(3, 3)

    for i, ax in enumerate(axes.flat):
        # plot img
        ax.imshow(images[i, :, :, :], interpolation='spline16')

        # show true & predicted classes
        cls_true_name = label_names[cls_true[i]]
        if cls_pred is None:
            xlabel = "{0} ({1})".format(cls_true_name, cls_true[i])
        else:
            cls_pred_name = label_names[cls_pred[i]]
            xlabel = "True: {0}\nPred: {1}".format(
                cls_true_name, cls_pred_name
            )
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


def list_to_string(list):
    """

    Parameters
    ----------
    list: list of network to transform into string for write it in a file

    Returns
    -------

    """
    string = ""
    print(list)
    for sub_list in range(len(list)):
        if sub_list == len(list)-1:
            if list[sub_list] is None:
                string += 'None'
            else:
                string += str(list[sub_list])
        else:
            for num in list[sub_list]:
                string += str(num)
                string += ","
    print(string)
    return string

def lists_to_list(list):
    """

    Parameters
    ----------
    list: lists of network to transform into list for write it in a file

    Returns
    -------

    """

    list2 = []
    for sub_list in range(len(list)):
        if sub_list == len(list)-1:
            list2.append(list[sub_list])
        else:
            for num in list[sub_list]:
                list2.append(str(num))

    return list2

def read_add_0(file):
    f = open(file, "r")
    f2 = open("e2epp.txt", "w")
    for line in f:
        list = line.split(",")
        for _ in range(len(list), 151):
            list.insert(len(list)-2,'0')

        f2.write(",".join(list))

    f.close()
    f2.close()

if __name__ == '__main__':
    read_add_0("e2eppDataPure.txt")


