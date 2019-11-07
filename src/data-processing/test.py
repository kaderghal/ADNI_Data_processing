
import io_data.data_acces_file as daf
import plot.plot_data as plot_data

l = "/home/karim/workspace/ADNI_workspace/results/ADNI_des/F_28P_F1_MS2_MB10D/HIPP/3D/AD-MCI/train/6_HIPP_alz_ADNI_1_train_AD-MCI_053_S_1044.pkl"

model = daf.read_model(l)
    
print(model.hippMetaDataVector)

plot_data.plot_ROI_all(model.hippRight, model.hippLeft, [[13, 16], [13, 16], [13, 16]], [[13, 16], [13, 16], [13, 16]])

    
    