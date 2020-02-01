const math = require('mathjs');

class RecommendModel {
    constructor(R, nf=200, alpha=40, lambda=40) {
        this.nu = R.length;
        this.ni = R[0].length;
        this.nf = nf;
        this.lambda = lambda;

        const X = math.random([this.nu, this.nf]);
        const Y = math.random([this.ni, this.nf]);
        this.X = X.map(function(array) {
            return array.map(x => x * 0.01);
        });
        this.Y = Y.map(function(array) {
            return array.map(y => y * 0.01);
        });

        this.P = R.map(function(array) {
            return array.map(function(i) {
                return (i > 0) ? 1 : 0;
            });
        });

        this.C = R.map(function(array) {
            return array.map(function(i) {
                return i * alpha + 1;
            });
        });
    }

    // loss() {
    //     const predicted = this.predict();
    // }

    optimize_user() {
        const yT = math.transpose(this.Y);
        const yTy = math.multiply(yT, this.Y);

        for(let u = 0; u < this.nu; ++u) {
            const Cu = math.diag(this.C[u]);
            const Cu_sub_I = math.subtract(Cu, math.identity(math.size(Cu)));
            const yTCuY = math.add(yTy, math.multiply(math.multiply(yT, Cu_sub_I), this.Y));
            const lI = math.identity(this.nf);
            for(let i = 0; i < this.nf; ++i) {
                lI._data[i][i] *= this.lambda;
            }
            const yTCuPu = math.multiply(math.multiply(yT, Cu), math.reshape(this.P[u], [this.ni, 1]));
            const inv_yTCuY_add_lI = math.inv(math.add(yTCuY, lI));

            this.X[u] = math.reshape(math.multiply(inv_yTCuY_add_lI, yTCuPu), [this.nf])._data;
        }
    }

    optimize_item() {
        const xT = math.transpose(this.X);
        const xTx = math.multiply(xT, this.X);

        let Y = [];
        for(let i = 0; i < this.ni; ++i) {
            const Ci = math.diag(math.squeeze(math.column(this.C, i)));
            const Ci_sub_I = math.subtract(Ci, math.identity(math.size(Ci)));
            const xTCiX = math.add(xTx, math.multiply(math.multiply(xT, Ci_sub_I), this.X));
            const lI = math.identity(this.nf);
            for(let i = 0; i < this.nf; ++i) {
                lI._data[i][i] *= this.lambda;
            }
            const xTCiPi = math.multiply(math.multiply(xT, Ci), math.reshape(math.column(this.P, i), [this.nu, 1]));
            const inv_xTCiX_add_lI = math.inv(math.add(xTCiX, lI));
            const Yi = math.reshape(math.multiply(inv_xTCiX_add_lI, xTCiPi), [1, this.nf]);
            Y = Y.concat(Yi._data);
        }
        this.Y = Y;
    }

    train_one_step() {
        this.optimize_user();
        this.optimize_item();
    }

    train(epoch=15) {
        for(let i = 0; i < epoch; ++i) {
            this.train_one_step();
            console.log('######### Step : %d #########', i + 1);
        }
    }

    predict() {
        return math.multiply(this.X, math.transpose(this.Y));
    }
}

module.exports = RecommendModel;