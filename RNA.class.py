import random, copy
import numpy as np

class RNA(object):
    def __init__(self, amostras, saida_desejada,taxa_aprendizado = 0.6,max_iteracoes = 100):
        #dados referentes ao treinamento
        self.amostras               = np.array(amostras)
        self.saida_desejada         = np.array(saida_desejada)
        self.taxa_aprendizado       = taxa_aprendizado
        self.max_iteracoes          = max_iteracoes
        #dados importantes da dimensão da RNA
        self.N_neuronios_hidden     = 4
        self.N_amostras             = len(amostras)
        self.N_entradas_amostra     = len(amostras[0])
        #saida camadas
        self.y_hidden               = []
        self.y_output               = []
        #pesos
        self.w1                     = []
        self.w2                     = []
        self.bias_hidden            = []
        self.bias_output            = []
        self.media_erros            = [] 
    
    
    #função de treinamento
    def treinar(self):
        #iniciando os valores do bias_hidden
        for i in range(self.N_neuronios_hidden):
            self.bias_hidden[i] = random.random()
        
        for i in range(len(self.saida_desejada[0])):
            self.bias_output[i] = random.random()

        #iniciando os pesos w_1 com valores aleatórios
        for i in range(self.N_entradas_amostra):
            for j in range(self.N_amostras):
                self.w1.append(random.random())
        
        #iniciando os pesos w_2 com valores aleatósigmoidx(y_hidden(i)-bias_hidden(i))ios
            for j in range(self.N_neuronios_hidden):
                self.w2.append(random.random())
        
        #dimensão dos vetores W1 e W2
        #W1[N_neuronios_hidden][N_entradas_amostra]
        #W2[1][N_neuronios_hidden]
        epoca       = 0
        media_erro  = []
        while True:
            erro        = False
            erro_local  = 0

            for index_amostra in range(self.N_amostras):
                #passar pela camada escondida
                for i in range(self.N_neuronios_hidden):
                    self.y_hidden[i] = 0
                    for j in range(self.N_entradas_amostra):
                        self.y_hidden[i] = self.y_hidden[i]+self.w1[i][j]*self.amostras[index_amostra][j]
                    #gerando saída na função sigmoid
                    self.y_hidden[i] = func_sigmoid(self.y_hidden[i]-self.bias_hidden[i])
                
                
                #passar pela camada de saida
                for i in range(len(self.saida_desejada[0])):
                    self.y_output[i] = 0
                    for j in range(self.N_neuronios_hidden):
                        self.y_output[i] = self.y_output[i] + self.w2[i][j]*self.y_hidden[j]
                    #gerando saída na função sigmoid
                    self.y_output[i] = func_sigmoid(self.y_output[i]-self.bias_output)
                
                if(self.y_output[0] != self.saida_desejada[index_amostra]):
                    #Backpropagation: Calculando erros

                    taxa_erro = self.saida_desejada[index_amostra] - self.y_output[0]

                    #somando erro para gerar gráficos posteriormente
                    erro_local += erro_local + abs(taxa_erro)

                    #descida de gradiente
                    delta_y_output = derivada_sigmoid(self.y_output[0])*taxa_erro

                    #calculando a correção dos pesos w2, 
                    # que ligam os neuronios da camada escondida
                    #com os neurnonios da camada de saída

                    #w2
                    correcao_w2 = []

                    for i in range(len(self.saida_desejada[0])):
                        for j in range(self.N_neuronios_hidden):
                            correcao_w2[i][j] = self.taxa_aprendizado*self.y_hidden[i]*delta_y_output
                        self.bias_output[i] = self.taxa_aprendizado*-1*delta_y_output
                    
                    #calculando erros da camada escondida
                    delta_y_hidden = []
                    for i in range(self.N_neuronios_hidden):
                        delta_y_hidden[i] = derivada_sigmoid(self.y_hidden[i]*delta_y_output*self.w2[0][i])

                    #calculando correção dos pesos w1
                    correcao_w1 = []
                    for i in range(self.N_neuronios_hidden):
                        for j in range(self.N_entradas_amostra):
                            correcao_w1[i][j] = self.taxa_aprendizado*self.amostras[index_amostra][j]*delta_y_hidden[i]

                    #******************************************************
                    # Backpropagation: aqui começa a atualização dos pesos
                    #******************************************************
                    #com os pesos w2 da camada de saída 
                    for i in range(len(self.saida_desejada[0])):
                        for j in range(self.N_neuronios_hidden):
                            self.w2[i][j] += correcao_w2[i][j]
                    
                    #com os pesos w1 da camada escondida
                    for i in range(self.N_neuronios_hidden):
                        for j in range(self.N_entradas_amostra):
                            self.w1[i][j] += correcao_w1[i][j]
                    
                    erro = True

            media_erro[epoca] = 0
            media_erro[epoca] = abs(erro_local/self.N_amostras)
            epoca += 1

            if(epoca ==  self.max_iteracoes or erro == False):
                print("Pesos w1: ")
                print(self.w1)
                print("Pesos w2: ")
                print(self.w2)
                print("Epoca: ",epoca)
                break

        #gerar gráfico de média erros                    



