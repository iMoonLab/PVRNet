from models import *
import config
import numpy as np



class PVRNet(nn.Module):
    def __init__(self, n_classes=40, init_weights=True):
        super(PVRNet, self).__init__()

        self.fea_dim = 1024
        self.num_bottleneck = 512
        self.n_scale = [2, 3, 4]
        self.n_neighbor = config.pv_net.n_neighbor

        self.mvcnn = BaseFeatureNet(base_model_name=config.base_model_name)

        # Point cloud net
        self.trans_net = transform_net(6, 3)
        self.conv2d1 = conv_2d(6, 64, 1)
        self.conv2d2 = conv_2d(128, 64, 1)
        self.conv2d3 = conv_2d(128, 64, 1)
        self.conv2d4 = conv_2d(128, 128, 1)
        self.conv2d5 = conv_2d(320, 1024, 1)

        self.fusion_fc_mv = nn.Sequential(
            fc_layer(4096, 1024, True),
        )

        self.fusion_fc = nn.Sequential(
            fc_layer(2048, 512, True),
        )

        self.fusion_conv1 = nn.Sequential(
            nn.Linear(2048, 1),
        )

        self.fusion_fc_scales = nn.ModuleList()

        for i in range(len(self.n_scale)):
            scale = self.n_scale[i]
            fc_fusion = nn.Sequential(
                        fc_layer((scale+1) * self.fea_dim, self.num_bottleneck, True),
                        )
            self.fusion_fc_scales += [fc_fusion]

        self.sig = nn.Sigmoid()

        self.fusion_mlp2 = nn.Sequential(
            fc_layer(1024, 256, True),
            nn.Dropout(p=0.5)
        )
        self.fusion_mlp3 = nn.Linear(256, n_classes)
        if init_weights:
            self.init_mvcnn()
            self.init_dgcnn()

    def init_mvcnn(self):
        print(f'init parameter from mvcnn {config.base_model_name}')
        mvcnn_state_dict = torch.load(config.view_net.ckpt_load_file)['model']
        pvrnet_state_dict = self.state_dict()

        mvcnn_state_dict = {k.replace('features', 'mvcnn', 1): v for k, v in mvcnn_state_dict.items()}
        mvcnn_state_dict = {k: v for k, v in mvcnn_state_dict.items() if k in pvrnet_state_dict.keys()}
        pvrnet_state_dict.update(mvcnn_state_dict)
        self.load_state_dict(pvrnet_state_dict)
        print(f'load ckpt from {config.view_net.ckpt_load_file}')

    def init_dgcnn(self):
        print(f'init parameter from dgcnn')
        dgcnn_state_dict = torch.load(config.pc_net.ckpt_load_file)['model']
        pvrnet_state_dict = self.state_dict()

        dgcnn_state_dict = {k: v for k, v in dgcnn_state_dict.items() if k in pvrnet_state_dict.keys()}
        pvrnet_state_dict.update(dgcnn_state_dict)
        self.load_state_dict(pvrnet_state_dict)
        print(f'load ckpt from {config.pc_net.ckpt_load_file}')


    def forward(self, pc, mv, get_fea=False):
        batch_size = pc.size(0)
        view_num = mv.size(1)
        mv, mv_view = self.mvcnn(mv)

        x_edge = get_edge_feature(pc, self.n_neighbor)
        x_trans = self.trans_net(x_edge)
        x = pc.squeeze(-1).transpose(2, 1)
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

        mv_view = mv_view.view(batch_size * view_num, -1)
        mv_view = self.fusion_fc_mv(mv_view)
        mv_view_expand = mv_view.view(batch_size, view_num, -1)

        pc = x5.squeeze()
        pc_expand = pc.unsqueeze(1).expand(-1, view_num, -1)
        pc_expand = pc_expand.contiguous().view(batch_size*view_num, -1)

        # Get Relation Scores
        fusion_mask = torch.cat((pc_expand, mv_view), dim=1)
        fusion_mask = self.fusion_conv1(fusion_mask)
        fusion_mask = fusion_mask.view(batch_size, view_num, -1)
        fusion_mask = self.sig(fusion_mask)

        # Rank Relation Scores
        mask_val, mask_idx = torch.sort(fusion_mask, dim=1, descending=True)
        mask_idx = mask_idx.expand(-1, -1, mv_view.size(-1))

        # Enhance View Feature
        mv_view_enhance = torch.mul(mv_view_expand, fusion_mask) + mv_view_expand

        # Get Point-Single-view Fusion
        fusion_global = self.fusion_fc(torch.cat((pc_expand, mv_view_enhance.view(batch_size*view_num, self.fea_dim)), dim=1))
        fusion_global, _ = torch.max(fusion_global.view(batch_size, view_num, self.num_bottleneck), dim=1)

        # Get Point-Multi-view Fusion
        scale_out = []
        for i in range(len(self.n_scale)):
            mv_scale_fea = torch.gather(mv_view_enhance, 1, mask_idx[:, :self.n_scale[i], :]).view(batch_size, self.n_scale[i]*self.fea_dim)
            mv_pc_scale = torch.cat((pc, mv_scale_fea), dim=1)
            mv_pc_scale = self.fusion_fc_scales[i](mv_pc_scale)
            scale_out.append(mv_pc_scale.unsqueeze(2))
        scale_out = torch.cat(scale_out, dim=2).mean(2)
        final_out = torch.cat((scale_out, fusion_global),1)

        # Final FC Layers
        net_fea = self.fusion_mlp2(final_out)
        net = self.fusion_mlp3(net_fea)

        if get_fea:
            return net, net_fea
        else:
            return net

