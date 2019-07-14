from models import *


class DGCNN(nn.Module):
    def __init__(self, n_neighbor=20, num_classes=20):
        super(DGCNN, self).__init__()
        self.n_neighbor = n_neighbor
        self.trans_net = transform_net(6, 3)
        self.conv2d1 = conv_2d(6, 64, 1)
        self.conv2d2 = conv_2d(128, 64, 1)
        self.conv2d3 = conv_2d(128, 64, 1)
        self.conv2d4 = conv_2d(128, 128, 1)
        self.conv2d5 = conv_2d(320, 1024, 1)
        self.mlp1 = nn.Sequential(
            fc_layer(1024, 512, True),
            nn.Dropout(p=0.5)
        )
        self.mlp2 = nn.Sequential(
            fc_layer(512, 256, True),
            nn.Dropout(p=0.5)
        )
        self.mlp3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x_edge = get_edge_feature(x, self.n_neighbor)
        x_trans = self.trans_net(x_edge)
        x = x.squeeze(-1).transpose(2, 1)
        x = torch.bmm(x, x_trans)
        x = x.transpose(2, 1)

        x1 = get_edge_feature(x, self.n_neighbor)
        x1 = self.conv2d1(x1)
        x1, _ = torch.max(x1, dim=-1, keepdim=True)

        x2 = get_edge_feature(x1, self.n_neighbor)
        x2 = self.conv2d2(x2)
        x2, _ = torch.max(x2, dim=-1, keepdim=True)

        x3 = get_edge_feature(x2, self.n_neighbor)
        x3 = self.conv2d3(x3)
        x3, _ = torch.max(x3, dim=-1, keepdim=True)

        x4 = get_edge_feature(x3, self.n_neighbor)
        x4 = self.conv2d4(x4)
        x4, _ = torch.max(x4, dim=-1, keepdim=True)

        x5 = torch.cat((x1, x2, x3, x4), dim=1)
        x5 = self.conv2d5(x5)
        x5, _ = torch.max(x5, dim=-2, keepdim=True)

        net = x5.view(x5.size(0), -1)
        net = self.mlp1(net)
        net = self.mlp2(net)
        net = self.mlp3(net)

        return net


