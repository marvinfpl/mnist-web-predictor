import model
import pandas as pd

print(__name__)

if __name__ == '__main__':

    """
        =======================================

                    HYPERPARAMETERS

        =======================================
    """
    in_ch = (1, 28, 28)
    out_ch = 4
    learning_rate = 0.002
    kernel_size = 3
    stride = 1
    padding = 1
    batch_size = 128
    epochs = 500
    max_grad_norm = 2.0


    """
        =======================================

                        MODEL

        =======================================
    """

    trainer = model.Trainer(in_ch, out_ch, learning_rate, batch_size, epochs, max_grad_norm, kernel_size, stride, padding)

    X_train, y_train, X_test, y_test = pd.read_csv("X_train"), pd.read_csv("y_train"), pd.read_csv("X_test"), pd.read_csv("y_test")

    trainer.learn(X_train, y_train)
    loss, report = trainer.evaluate(X_test, y_test)

    print(f" Loss {loss}, sklearn report: {report}")

    trainer.save("/model")

