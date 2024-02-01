import SimpleITK as sitk
import numpy as np
import logging

logger = logging.getLogger(__name__)

from custom_algorithm import Hanseg2023Algorithm

LABEL_dict = {
    "background": 0,
    "A_Carotid_L": 1,
    "A_Carotid_R": 2,
    "Arytenoid": 3,
    "Bone_Mandible": 4,
    "Brainstem": 5,
    "BuccalMucosa": 6,
    "Cavity_Oral": 7,
    "Cochlea_L": 8,
    "Cochlea_R": 9,
    "Cricopharyngeus": 10,
    "Esophagus_S": 11,
    "Eye_AL": 12,
    "Eye_AR": 13,
    "Eye_PL": 14,
    "Eye_PR": 15,
    "Glnd_Lacrimal_L": 16,
    "Glnd_Lacrimal_R": 17,
    "Glnd_Submand_L": 18,
    "Glnd_Submand_R": 19,
    "Glnd_Thyroid": 20,
    "Glottis": 21,
    "Larynx_SG": 22,
    "Lips": 23,
    "OpticChiasm": 24,
    "OpticNrv_L": 25,
    "OpticNrv_R": 26,
    "Parotid_L": 27,
    "Parotid_R": 28,
    "Pituitary": 29,
    "SpinalCord": 30,
}


def normalize_3d_image(image):
    image_array = sitk.GetArrayFromImage(image)

    normalized_image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))

    normalized_image = sitk.GetImageFromArray(normalized_image_array)
    normalized_image.CopyInformation(image)  

    return normalized_image

def resize_and_resample(image, new_size, interpolator=sitk.sitkLinear,normalize=False):

    if normalize == True:
        n_image = normalize_3d_image(image)
    else:
        n_image = image
    original_size = n_image.GetSize()
    original_spacing = n_image.GetSpacing() 

    new_spacing = [old_sz * old_spc / new_sz for old_sz, old_spc, new_sz in zip(original_size, original_spacing, new_size)]

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetInterpolator(interpolator)

    return resampler.Execute(n_image)

class Residual_Block(tc.nn.Module):
    def __init__(self, input_size, output_size):
        super(Residual_Block, self).__init__()

        self.module = tc.nn.Sequential(                                         
           tc.nn.Conv3d(input_size, output_size, 3, stride=1, padding=1),        
           tc.nn.GroupNorm(output_size, output_size),                           
           tc.nn.LeakyReLU(0.01, inplace=True),                                 
           tc.nn.Conv3d(output_size, output_size, 3, stride=1, padding=1),      
           tc.nn.GroupNorm(output_size, output_size),                           
           tc.nn.LeakyReLU(0.01, inplace=True),
        )

        self.conv = tc.nn.Sequential(
            tc.nn.Conv3d(input_size,output_size,1)
        )

    def forward(self,x):
        return self.module(x) + self.conv(x)  

class Res_Unet_16(tc.nn.Module):
    def __init__(self, filters=[16, 32, 64, 128, 256, 512], num_classes=31,modalities=2):
        super(Res_Unet_16, self).__init__()

        self.encoder_1 = tc.nn.Sequential(
            Residual_Block(modalities,filters[0]),
            tc.nn.Conv3d(filters[0],filters[0],3,stride=1,padding=1),
            tc.nn.GroupNorm(filters[0],filters[0]),
            tc.nn.LeakyReLU(0.01, inplace=True)
        )
        self.encoder_2 = tc.nn.Sequential(
            tc.nn.MaxPool3d(2,stride=2),
            Residual_Block(filters[0],filters[1]),
            tc.nn.Conv3d(filters[1],filters[1],kernel_size=3,stride=1,padding=1),
            tc.nn.GroupNorm(filters[1],filters[1]),
            tc.nn.LeakyReLU(0.01, inplace=True)
        )
        self.encoder_3 = tc.nn.Sequential(
            tc.nn.MaxPool3d(2,stride=2),
            Residual_Block(filters[1],filters[2]),
            tc.nn.Conv3d(filters[2],filters[2],kernel_size=3,stride=1,padding=1),
            tc.nn.GroupNorm(filters[2],filters[2]),
            tc.nn.LeakyReLU(0.01, inplace=True)
        )
        self.encoder_4 = tc.nn.Sequential(
            tc.nn.MaxPool3d(2,stride=2),
            Residual_Block(filters[2],filters[3]),
            tc.nn.Conv3d(filters[3],filters[3],kernel_size=3,stride=1,padding=1),
            tc.nn.GroupNorm(filters[3],filters[3]),
            tc.nn.LeakyReLU(0.01,inplace=True)
        )
        self.encoder_5 = tc.nn.Sequential(
            tc.nn.MaxPool3d(2,stride=2),
            Residual_Block(filters[3],filters[4]),
            tc.nn.Conv3d(filters[4],filters[4],kernel_size=3,stride=1,padding=1),
            tc.nn.GroupNorm(filters[4],filters[4]),
            tc.nn.LeakyReLU(0.01,inplace=True),
            tc.nn.ConvTranspose3d(filters[4],filters[3],kernel_size=2,stride=2)
        )
        self.decoder_4 = tc.nn.Sequential(
            Residual_Block(filters[4],filters[3]),
            tc.nn.Conv3d(filters[3],filters[3],kernel_size=3,stride=1,padding=1),
            tc.nn.GroupNorm(filters[3],filters[3]),
            tc.nn.LeakyReLU(0.01,inplace=True),
            tc.nn.ConvTranspose3d(filters[3],filters[2],kernel_size=2,stride=2)
        )
        self.decoder_3 = tc.nn.Sequential(
            Residual_Block(filters[3],filters[2]),
            tc.nn.Conv3d(filters[2],filters[2],kernel_size=3,stride=1,padding=1),
            tc.nn.GroupNorm(filters[2],filters[2]),
            tc.nn.LeakyReLU(0.01,inplace=True),
            tc.nn.ConvTranspose3d(filters[2],filters[1],kernel_size=2,stride=2)
        )
        self.decoder_2 = tc.nn.Sequential(
            Residual_Block(filters[2],filters[1]),
            tc.nn.Conv3d(filters[1],filters[1],kernel_size=3,stride=1,padding=1),
            tc.nn.GroupNorm(filters[1],filters[1]),
            tc.nn.LeakyReLU(0.01,inplace=True),
            tc.nn.ConvTranspose3d(filters[1],filters[0],kernel_size=2,stride=2)
        )
        self.decoder_1 = tc.nn.Sequential(
            Residual_Block(filters[1],filters[0]),
            tc.nn.Conv3d(filters[0],filters[0],kernel_size=3,stride=1,padding=1),
            tc.nn.GroupNorm(filters[0],filters[0]),
            tc.nn.ConvTranspose3d(filters[0],num_classes,kernel_size=1)
        )

    def forward(self,x):
        x1 = self.encoder_1(x)
        x2 = self.encoder_2(x1)
        x3 = self.encoder_3(x2)
        x4 = self.encoder_4(x3)
        x5 = self.encoder_5(x4)
        
        d4 = self.decoder_4(tc.cat((x4, x5), dim=1))
        d3 = self.decoder_3(tc.cat((x3, d4), dim=1))
        d2 = self.decoder_2(tc.cat((x2, d3), dim=1))
        d1 = self.decoder_1(tc.cat((x1, d2), dim=1))
        return d1

class MyHanseg2023Algorithm(Hanseg2023Algorithm):
    def __init__(self):
        super().__init__()

    def predict(self, *, image_ct: sitk.Image, image_mrt1: sitk.Image) -> sitk.Image:
        
        old_size = image_ct.GetSize()
        new_size = (512,512,224)
        new_ct = resize_and_resample(image_ct,new_size=new_size,normalize=True)
        new_mr = resize_and_resample(image_mrt1,new_size=new_size,normalize=True)
        ct_array = sitk.GetArrayFromImage(new_ct)
        mr_array = sitk.GetArrayFromImage(new_mr)
        dual_array = np.stack((ct_array,mr_array),axis=0)
        input_tensor = tc.from_numpy(dual_array)
        del dual_array
        del mr_array
        del ct_array
        del new_ct
        del new_mr
        del new_size
        device = tc.device('cuda' if tc.cuda.is_available() else 'cpu')
        model = Res_Unet_16().to(device)
        patch_size = (192,224,224)
        subject = tio.Subject(dual = tio.ScalarImage(tensor=input_tensor))
        model.load_state_dict(tc.load('192_224_fz_sb16_N_latest.pth',map_location=tc.device(device)))
        model.eval()
        with tc.no_grad():
            patch_overlap = 0
            grid_sampler = tio.inference.GridSampler(subject,patch_size,patch_overlap)
            patch_loader = tc.utils.data.DataLoader(grid_sampler, batch_size=1)
            aggregator = tio.inference.GridAggregator(grid_sampler)  
            for patches_batch in patch_loader:
                input_tensor = patches_batch['dual'][tio.DATA].type(tc.FloatTensor).to(device)
                locations = patches_batch[tio.LOCATION]
                outputs = model(input_tensor)
                aggregator.add_batch(outputs, locations)  
            output_tensor = aggregator.get_output_tensor().detach().cpu()
        #print('Output tensor created')
        output = np.zeros(output_tensor[0].shape)
        input = output_tensor.numpy()
        for i in range(input.shape[0]):
            mask = input[i] >= 0.5
            output[mask] = i
        
        output_image = sitk.GetImageFromArray(output)
        os_output_image = resize_and_resample(output_image,new_size=old_size,interpolator=sitk.sitkNearestNeighbor)
        os_output_image.CopyInformation(image_ct)
        #print('Final post-processing accomplished')
        output_seg = sitk.Cast(os_output_image, sitk.sitkUInt8)
        return output_seg

if __name__ == "__main__":
    MyHanseg2023Algorithm().process()