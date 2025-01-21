import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping


def plot_training_fluctuation(m3_history, m4_history, e_history):
    m3_acc = m3_history.history['accuracy']
    m4_acc = m4_history.history['accuracy']
    e_acc = e_history.history['accuracy']
    
    epochs = range(1, len(m4_acc) + 1)
    plt.plot(epochs, m3_acc, 'bo-', label='m3 training acc')
    plt.plot(epochs, m4_acc, 'ro-', label='m4 training acc')
    plt.plot(epochs, e_acc, 'yo-', label='e training acc')

    plt.title('Local Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    
    
def evaulate_acc(model, m4_tr_x, m4_tr_y):
    y_pred = model.predict(m4_tr_x)
    y_final = (y_pred > 0.5).astype(int).reshape(
        m4_tr_x.shape[0])
    
    m4_tr_y2 = []
    for ind in m4_tr_y:
        m4_tr_y2.append(ind[0])
    m4_tr_y3 = np.array(m4_tr_y2)
    
    total_len = len(m4_tr_y3)
    differences = np.isclose(y_final, m4_tr_y3)
    count_true_in_difference = np.sum(differences)
    rate_of_correct_prediction = (
        count_true_in_difference/total_len)*100
    
    return round(rate_of_correct_prediction, 2) 


def plot_training_acc(total_rounds, m4_acc_ls, m3_acc_ls, e_acc_ls):
    comm_round_ls = list(range(1, total_rounds + 1))
    
    plt.plot(comm_round_ls, m4_acc_ls, 
             'r-o', label='m4 training set acc')
    plt.plot(comm_round_ls, m3_acc_ls, 
             'b-o', label='m3 training set acc')
    plt.plot(comm_round_ls, e_acc_ls, 
             'y-o', label='e training set acc')
    
    plt.xticks(np.arange(1, total_rounds+1, 5))
    plt.title('Global Model Accuracy')
    plt.xlabel('Communication Round')
    plt.ylabel('Accuracy')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.show()


def three_model_train_round(
    model_m3, model_m4, model_e,   
    m3_tr_x, m3_tr_y, m4_tr_x, m4_tr_y, e_tr_x, e_tr_y,
    epo_num, bs):
    
    callback = EarlyStopping(monitor='loss', 
                             patience=epo_num, 
                             restore_best_weights=True)

    m3_h = model_m3.fit(m3_tr_x, m3_tr_y, 
                        batch_size = bs, epochs = epo_num, 
                        verbose = 0, callbacks=[callback])
    
    m4_h = model_m4.fit(m4_tr_x, m4_tr_y, 
                        batch_size = bs, epochs = epo_num, 
                        verbose = 0, callbacks=[callback])
    
    e_h = model_e.fit(e_tr_x, e_tr_y, 
                        batch_size = bs, epochs = epo_num, 
                        verbose = 0, callbacks=[callback])    

    return model_m3, model_m4, model_e











    