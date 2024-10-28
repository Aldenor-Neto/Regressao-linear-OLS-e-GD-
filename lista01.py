import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Carregar o arquivo CSV
data = pd.read_csv('artificial1d.csv', header=None, names=['x', 'y'])
x = data['x'].values
y = data['y'].values


# Função para cálculo do MSE
def calcular_mse(y, y_pred):
    return np.mean((y - y_pred) ** 2)


# Função para regressão com Mínimos Quadrados Ordinários (OLS)
def regressao_ols(x, y):
    X = np.vstack((np.ones(len(x)), x)).T  # Adiciona a coluna de 1's
    w = np.linalg.inv(X.T @ X) @ X.T @ y  # Calcula os parâmetros
    w0, w1 = w[0], w[1]
    y_pred = w0 + w1 * x
    mse = calcular_mse(y, y_pred)
    return w0, w1, mse, y_pred


# Função para Regressão Linear com Gradiente Descendente (salvando imagens a cada 10 épocas)
def gradiente_descendente(x, y):
    w0 = 0  # Inicialização do parâmetro w0
    w1 = 0  # Inicialização do parâmetro w1
    alpha = 0.01  # Taxa de aprendizado fixa
    epochs = 1000  # Número máximo de épocas fixo
    tol = 1e-3  # Tolerância fixa
    mse_values = []

    # Diretório para salvar as imagens do gradiente descendente
    gif_dir = os.path.join(output_dir, 'imagens_gif')
    if not os.path.exists(gif_dir):
        os.makedirs(gif_dir)

    for t in range(epochs):
        y_pred = w0 + w1 * x
        error = y - y_pred
        w0 += alpha * np.mean(error)
        w1 += alpha * np.mean(error * x)

        mse = calcular_mse(y, y_pred)
        mse_values.append(mse)

        # Salvar a primeira época, a cada 10 épocas, e a última época
        if t == 0 or t % 10 == 0 or t == epochs - 1:
            plt.figure()
            plt.scatter(x, y, color='blue', label='Dados')
            plt.plot(x, y_pred, color='red', linestyle='--', label=f'Época {t}')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.legend()
            plt.title(f'Gradiente Descendente - Época {t}, MSE: {mse:.5f}')
            plt.savefig(os.path.join(gif_dir, f'gradiente_descendente_epoca_{str(t).zfill(3)}.png'))
            plt.close()

        # Critério de convergência
        if t > 0 and abs(mse_values[-2] - mse_values[-1]) < tol:
            print(f'Convergência atingida na época {t}.')
            break

    return w0, w1, mse_values, y_pred

# Exibição dos resultados da Regressão Linear via OLS
w0_ols, w1_ols, mse_ols, y_pred_ols = regressao_ols(x, y)
print("Regressão Linear via OLS:")
print(f"Parâmetro w0 (intercepto): {w0_ols}")
print(f"Parâmetro w1 (coeficiente): {w1_ols}")
print(f"MSE (Erro Quadrático Médio): {mse_ols}")

# Verifica se o diretório 'imagens' existe, se não, cria
output_dir = 'imagens'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Plot do gráfico para OLS
plt.figure()
plt.scatter(x, y, color='blue', label='Dados')
plt.plot(x, y_pred_ols, color='green', linestyle='--', label='Regressão OLS')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title(f'Regressão Linear - OLS, MSE: {mse_ols:.5f}')
plt.savefig(os.path.join(output_dir, 'regressao_ols.png'))
plt.close()

# Exibição dos resultados da Regressão Linear via Gradiente Descendente
w0_gd, w1_gd, mse_values_gd, y_pred_gd = gradiente_descendente(x, y)
print("\nRegressão Linear via Gradiente Descendente:")
print(f"Parâmetro w0 (intercepto): {w0_gd}")
print(f"Parâmetro w1 (coeficiente): {w1_gd}")
print(f"Último MSE (Erro Quadrático Médio): {mse_values_gd[-1]}")

# Plot do gráfico para Gradiente Descendente
plt.figure()
plt.scatter(x, y, color='blue', label='Dados')
plt.plot(x, y_pred_gd, color='red', linestyle='--', label='Regressão GD')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title(f'Regressão Linear - Gradiente Descendente, MSE: {mse_values_gd[-1]:.5f}')
plt.savefig(os.path.join(output_dir, 'regressao_gd.png'))
plt.close()

# Plot da curva de aprendizagem (MSE ao longo das épocas para o GD)
plt.figure()
plt.plot(mse_values_gd, color='purple')
plt.xlabel('Épocas')
plt.ylabel('MSE')
plt.title('Curva de Aprendizagem - Gradiente Descendente')
plt.savefig(os.path.join(output_dir, 'curva_aprendizagem_gd.png'))
plt.close()  # Fecha a figura para liberar memória
