const tf = require('@tensorflow/tfjs-node');
const {inv} = require('mathjs');

class RecommendModel {
    constructor(rated_matrix, nf=200, alpha=40, lambda=40) {
        const R = tf.tensor2d(rated_matrix);
        this.nu = R.shape[0];
        this.ni = R.shape[1];
        this.nf = nf;
        this.lambda = lambda;

        const init_X = tf.randomNormal([this.nu, this.nf]).mul(0.01);
        const init_Y = tf.randomNormal([this.ni, this.nf]).mul(0.01);
        this.X = tf.variable(init_X);
        this.Y = tf.variable(init_Y);

        const init_P = rated_matrix.map(function (array) {
            return array.map(function (i) {
                return (i > 0) ? 1 : 0;
            });
        });
        const init_C = R.mul(alpha).add(1);
        this.P = tf.variable(tf.tensor2d(init_P));
        this.C = tf.variable(init_C);
    }

    loss() {
        const predicted = this.predict();
        const predict_error = tf.sum(tf.mul(this.C, tf.square(tf.sub(this.P, predicted))));
        const regularization = tf.add(tf.sum(tf.square(this.X)), tf.sum(tf.square(this.Y))).mul(this.lambda);
        const total_loss = tf.add(predict_error, regularization);

        return {
            predict_error: predict_error,
            regularization: regularization,
            total_loss: total_loss
        };
    }

    optimize_user() {
        const yT = tf.transpose(this.Y);
        const yTy = tf.matMul(yT, this.Y);

        let X;
        for(let u = 0; u < this.nu; ++u) {
            const Cu = tf.diag(this.C.gather(u));
            const yTCuY = tf.add(yTy, tf.matMul(tf.matMul(yT, Cu.sub(tf.eye(Cu.shape[1]))), this.Y));
            const lI = tf.eye(this.nf).mul(this.lambda);
            const yTCuPu = tf.matMul(tf.matMul(yT, Cu), tf.expandDims(this.P.gather(u), 1));

            let tmp = tf.add(yTCuY, lI).arraySync();
            tmp = tf.tensor(inv(tmp));
            const Xu = tf.reshape(tf.matMul(tmp, yTCuPu), [1, this.nf]);

            if(u === 0) X = tf.clone(Xu);
            else        X = tf.concat([X, Xu]);
        }
        this.X.assign(X);
    }

    optimize_item() {
        const xT = tf.transpose(this.X);
        const xTx = tf.matMul(xT, this.X);

        let Y;
        for(let i = 0; i < this.ni; ++i) {
            const Ci = tf.diag(tf.transpose(this.C).gather(i));
            const xTCiX = xTx.add(tf.matMul(tf.matMul(xT, Ci.sub(tf.eye(Ci.shape[1]))), this.X));
            const lI = tf.eye(this.nf).mul(this.lambda);
            const xTCiPi = tf.matMul(tf.matMul(xT, Ci), tf.expandDims(tf.transpose(this.P).gather(i), 1));

            let tmp = tf.add(xTCiX, lI).arraySync();
            tmp = tf.tensor(inv(tmp));
            const Yi = tf.reshape(tf.matMul(tmp, xTCiPi), [1, this.nf]);

            if(i === 0) Y = tf.clone(Yi);
            else        Y = tf.concat([Y, Yi]);
        }
        this.Y.assign(Y);
    }

    train_one_step() {
        this.optimize_user();
        this.optimize_item();
    }

    train(epoch=15) {
        const predict_errors = [];
        const regularization_list = [];
        const total_losses = [];

        for(let i = 0; i < epoch; ++i) {
            this.train_one_step();
            const obj = this.loss();
            predict_errors.push(obj.predict_error);
            regularization_list.push(obj.regularization);
            total_losses.push(obj.total_loss);

            console.log('##### Step : ' + (i+1) + ' #####');
            console.log('Predict error : ' + obj.predict_error);
            console.log('Regularization: ' + obj.regularization);
            console.log('Total loss: ' + obj.total_loss);
        }

        return {
            predict_errors: predict_errors,
            regularization_list: regularization_list,
            total_losses: total_losses
        }
    }

    predict() {
        return tf.matMul(this.X, tf.transpose(this.Y));
    }
}

module.exports = RecommendModel;