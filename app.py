import streamlit as st

def main():
    """ ### AHC016
        期間 2022/11/11~20 (終了) """

    ### memo
    ## 誤り訂正をせずに送信情報を予測するような問題と予想(違った).
    ## 最低限のbit数(graph_0なら 0個bitを立てる)でばらつき具合を確認してみる. -> G1配列.
    ## 非線形回帰で求めた近似式をグラフ生成時に参照して、bitを立てる数に重みづけをしたい.-> G2配列.
    ## htmlに埋め込むなら入力を受け取る方法を変えることと、図を表示するたびにキャッシュの削除(plt.clf())をしたほうがよさげ.

    ### imports
    import random
    from pprint import pprint
    import numpy as np
    import matplotlib.pyplot as plt

    ### functions
    def onefill(n:int) -> int:
        """ 1埋めする数を計算して返す.\n
            ここに近似式を取り入れたい. """
        return int((n/(168/(EPS/0.01)))**2)

    def make_h(bools:str) -> str:
        """ bool文字列の各要素を確率を基に反転して返す. """
        b = (0, 1)
        w = (100-EPS, EPS)
        l = [i for i in bools]
        for i in range(len(l)):
            c = random.choices(b, weights=w)[0]
            if c: l[i] = '0' if l[i]=='1' else '1'
        return ''.join(l)

    def show_3d_graph():
        """ 3dグラフを表示. """
        from mpl_toolkits.mplot3d import Axes3D ## 明示的には使ってないけど入れないとエラー

        ## データ.
        ## np.linspace()で各軸のポイントを100個に増やして滑らかにする.
        ##  -> float型なのでそのままではグラフの添え字として使えない.これをどうにかする必要がある.
        # x = np.linspace(0,M,100)
        # y = np.linspace(0,N1+1,100)
        x = list(range(M))
        y = list(range(N1+1))
        X, Y = np.meshgrid(x, y)
        Z = np.array(result_G1)

        ## 3d図を表示.
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.plot_surface(X, Y, Z, alpha=0.8)
        ax.set_xlabel('graph_index')
        ax.set_ylabel('bits')
        ax.set_zlabel('amount')
        plt.title('AHC016')
        st.pyplot(fig)

    def show_bar_graph():
        """ 棒グラフを表示. """
        ## データ.
        x = list(range(N1+1))
        y = result_G1[0]

        x_position = np.arange(len(x))

        plt.bar(x_position, y, tick_label=x)
        plt.xlabel('bits')
        plt.ylabel('amount')
        plt.title('AHC016')
        plt.show()

    def show_2d_graph():
        """ 近似曲線を表示. """
        ## データ.
        x_observed = list(range(N1+1))
        y_observed = result_G1[0]
        ## フィッティングカーブ用のポイント.
        x_latent = np.linspace(min(x_observed), max(x_observed), 100)
        ## 1次式から9次式までフィッティングしてみる.
        cf1 = ["最小2乗法（1次式）", lambda x, y: np.polyfit(x, y, 1)]
        cf2 = ["最小2乗法（2次式）", lambda x, y: np.polyfit(x, y, 2)]
        cf3 = ["最小2乗法（3次式）", lambda x, y: np.polyfit(x, y, 3)]
        cf4 = ["最小2乗法（4次式）", lambda x, y: np.polyfit(x, y, 4)]
        cf5 = ["最小2乗法（5次式）", lambda x, y: np.polyfit(x, y, 5)]
        cf6 = ["最小2乗法（6次式）", lambda x, y: np.polyfit(x, y, 6)]
        cf7 = ["最小2乗法（7次式）", lambda x, y: np.polyfit(x, y, 7)]
        cf8 = ["最小2乗法（8次式）", lambda x, y: np.polyfit(x, y, 8)]
        cf9 = ["最小2乗法（9次式）", lambda x, y: np.polyfit(x, y, 9)]

        ## pyscriptではsympyは使えないのでいったんコメントアウト.
        # import sympy as sym
        # from sympy.plotting import plot
        # sym.init_printing(use_unicode=True)
        # x, y = sym.symbols("x y")

        for method_name, method in [cf1, cf2, cf3, cf4, cf5, cf6, cf7, cf8, cf9]:
            print(method_name)
            # ## 係数の計算. sympy絡みのコード.
            # coefficients = method(x_observed, y_observed)
            # ## sympy を用いた数式の表示.
            # expr = 0
            # for index, coefficient in enumerate(coefficients):
            #     expr += coefficient * x ** (len(coefficients) - index - 1)
            # print(sym.Eq(y, expr))

            ## プロットと曲線の表示.
            fitted_curve = np.poly1d(method(x_observed, y_observed))(x_latent)
            print(fitted_curve)
            plt.scatter(x_observed, y_observed, label="observed")
            plt.plot(x_latent, fitted_curve, c="red", label="fitted")
            plt.grid()
            plt.legend()
            plt.show()

    # streamlit
    st.sidebar.title('データの設定')
    input_x=st.sidebar.number_input('グラフ数：',10,100,10)
    input_y=st.sidebar.number_input('エラー率：',0,40,0)
    input_z=st.sidebar.number_input('サンプルサイズ：',0,10000,0)

    ### inputs
    M = int(input_x)
    EPS = int(input_y)
    SAMPLE_SIZE = int(input_z)

    ### constant numbers
    random.seed(0)
    N1 = M-1        ## 最低限のビット数
    # N2 = M*(M-1)//2 ## 設問通りのビット数

    ### main
    G1 = ['1'*i + '0'*(N1-i) for i in range(M)]
    # G2 = ['1'*i + '0'*(N2-i) for i in range(M)]
    result_G1 = []
    # result_G2 = []
    for i in range(M):
        counter_G1 = [0 for _ in range(N1+1)]
        for _ in range(SAMPLE_SIZE):
            H = make_h(G1[i])
            counter_G1[H.count('1')] += 1
        result_G1.append(counter_G1)

    # print(*G1, sep='\n')
    # pprint(result_G1)

    # show_bar_graph()
    # show_2d_graph()
    show_3d_graph()

if __name__ == '__main__':
    main()