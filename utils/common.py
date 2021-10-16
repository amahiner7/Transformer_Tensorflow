import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

    return elapsed_mins, elapsed_secs


# 정확도 계산 함수
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def metric_batch(preds, labels):
    prediction = preds.argmax(1, keepdim=True)
    corrects = prediction.eq(labels.view_as(prediction)).sum().item()
    return corrects


# 시간 표시 함수
def format_time(elapsed):
    # 반올림
    elapsed_rounded = int(round(elapsed))

    # hh:mm:ss으로 형태 변경
    return str(datetime.timedelta(seconds=elapsed_rounded))


def display_attention(sentence, translation, attention, n_heads=8, n_rows=4, n_cols=2):

    assert n_rows * n_cols == n_heads

    # 출력할 그림 크기 조절
    fig = plt.figure(figsize=(15, 25))

    for i in range(n_heads):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)

        # 어텐션(Attention) 스코어 확률 값을 이용해 그리기
        _attention = attention.squeeze(0)[i].cpu().detach().numpy()

        cax = ax.matshow(_attention, cmap='bone')

        ax.tick_params(labelsize=12)
        ax.set_xticklabels([''] + ['<sos>'] + [t.lower() for t in sentence] + ['<eos>'], rotation=45)
        ax.set_yticklabels([''] + translation)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.close()


def display_loss(history):
    train_loss = history['loss']
    val_loss = history['val_loss']
    learning_rate = history['learning_rate']

    # 그래프로 표현
    x_len = np.arange(len(train_loss))
    plt.figure()
    plt.plot(x_len, val_loss, marker='.', c="blue", label='Validation loss')
    plt.plot(x_len, train_loss, marker='.', c="red", label='Train loss')
    # 그래프에 그리드를 주고 레이블을 표시
    plt.legend(loc='upper right')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss')
    plt.show()

    plt.clf()
    plt.figure()
    plt.plot(x_len, learning_rate, marker='.', c="yellow", label='Learning rate')
    plt.legend(loc='upper right')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('Learning rate')
    plt.title('Learning rate')
    plt.show()