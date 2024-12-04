import torch.nn as nn


def configuration(choice: int):
    # Overall 30 choices (from 0 to 29)
    activation = "relu" if choice % 2 == 0 else "sigmoid"
    choice = choice // 2

    layers = []
    if choice == 0:
        layers = nn.ModuleList(
            [nn.Linear(28 * 28, 128), nn.Linear(128, 64), nn.Linear(64, 10)]
        )
    elif choice == 1:
        layers = nn.ModuleList(
            [
                nn.Linear(28 * 28, 256),
                nn.Linear(256, 128),
                nn.Linear(128, 64),
                nn.Linear(64, 10),
            ]
        )
    elif choice == 2:
        layers = nn.ModuleList(
            [
                nn.Linear(28 * 28, 512),
                nn.Linear(512, 256),
                nn.Linear(256, 128),
                nn.Linear(128, 10),
            ]
        )
    elif choice == 3:
        layers = nn.ModuleList(
            [nn.Linear(28 * 28, 64), nn.Linear(64, 32), nn.Linear(32, 10)]
        )
    elif choice == 4:
        layers = nn.ModuleList(
            [
                nn.Linear(28 * 28, 128),
                nn.Linear(128, 128),
                nn.Linear(128, 64),
                nn.Linear(64, 10),
            ]
        )
    elif choice == 5:
        layers = nn.ModuleList(
            [
                nn.Linear(28 * 28, 512),
                nn.Linear(512, 256),
                nn.Linear(256, 128),
                nn.Linear(128, 64),
                nn.Linear(64, 10),
            ]
        )
    elif choice == 6:
        layers = nn.ModuleList(
            [
                nn.Linear(28 * 28, 128),
                nn.Linear(128, 128),
                nn.Linear(128, 128),
                nn.Linear(128, 64),
                nn.Linear(64, 10),
            ]
        )
    elif choice == 7:
        layers = nn.ModuleList(
            [
                nn.Linear(28 * 28, 64),
                nn.Linear(64, 64),
                nn.Linear(64, 64),
                nn.Linear(64, 64),
                nn.Linear(64, 10),
            ]
        )
    elif choice == 8:
        layers = nn.ModuleList(
            [
                nn.Linear(28 * 28, 1024),
                nn.Linear(1024, 512),
                nn.Linear(512, 256),
                nn.Linear(256, 10),
            ]
        )
    elif choice == 9:
        layers = nn.ModuleList(
            [nn.Linear(28 * 28, 32), nn.Linear(32, 16), nn.Linear(16, 10)]
        )
    elif choice == 10:
        layers = nn.ModuleList(
            [
                nn.Linear(28 * 28, 256),
                nn.Linear(256, 128),
                nn.Linear(128, 128),
                nn.Linear(128, 64),
                nn.Linear(64, 10),
            ]
        )
    elif choice == 11:
        layers = nn.ModuleList(
            [
                nn.Linear(28 * 28, 128),
                nn.Linear(128, 64),
                nn.Linear(64, 32),
                nn.Linear(32, 10),
            ]
        )
    elif choice == 12:
        layers = nn.ModuleList(
            [
                nn.Linear(28 * 28, 256),
                nn.Linear(256, 128),
                nn.Linear(128, 64),
                nn.Linear(64, 32),
                nn.Linear(32, 10),
            ]
        )
    elif choice == 13:
        layers = nn.ModuleList(
            [
                nn.Linear(28 * 28, 128),
                nn.Linear(128, 64),
                nn.Linear(64, 64),
                nn.Linear(64, 64),
                nn.Linear(64, 10),
            ]
        )
    elif choice == 14:
        layers = nn.ModuleList(
            [
                nn.Linear(28 * 28, 128),
                nn.Linear(128, 128),
                nn.Linear(128, 64),
                nn.Linear(64, 32),
                nn.Linear(32, 10),
            ]
        )

    return layers, activation
