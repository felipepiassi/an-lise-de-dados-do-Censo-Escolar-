import pandas as pd
import matplotlib.pyplot as plt
import os
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from sklearn.cluster import KMeans

# ============================
# CONFIGURAÇÃO DE CAMINHOS
# ============================

CSV_PATH = r"END. TABELA EM CSV "
OUTPUT_DIR = r"END. RELATORIOS"
os.makedirs(OUTPUT_DIR, exist_ok=True)
PDF_PATH = os.path.join(OUTPUT_DIR, "relatorio_censo_escolar.pdf")

# ============================
#  PREPARAÇÃO DOS DADOS
# ============================

def carregar_e_limpar_dados():
    """
    Carrega a planilha, renomeia as colunas e limpa os dados para análise.
    """
    print("\n🔄 Carregando e limpando os dados...")
    df = pd.read_csv(CSV_PATH, sep=';', skiprows=[1], encoding='latin1')

    # PANDAS
    # Renomear colunas para facilitar o acesso
    df.columns = [     # altere os dados da tabela para fazer outras analises
        'Localizacao', 'Creche_Parcial', 'Creche_Integral', 'Pre_escola_Parcial',
        'Pre_escola_Integral', 'Anos_Iniciais_Parcial', 'Anos_Iniciais_Integral',
        'Anos_Finais_Parcial', 'Anos_Finais_Integral', 'Medio_Parcial',
        'Medio_Integral', 'EJA_Fundamental', 'EJA_Medio'
    ]

    # Remover linhas de subtotais e de locais específicos
    df = df[~df['Localizacao'].isin(['BRASIL', 'Estadual e Municipal', 'Estadual', 'Municipal', 'Privada'])]  # altere os dados da tabela para fazer outras analises
    df.dropna(subset=['Localizacao'], inplace=True)
    df = df[df['Localizacao'] != '']
    df.fillna(0, inplace=True)

    # Converter colunas numéricas
    numeric_cols = df.columns[1:]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False), errors='coerce')
    
    df.dropna(subset=numeric_cols, inplace=True)
    
    print("✅ Dados limpos e prontos para análise.")
    return df

# ============================
# ETAPA 2 - ANÁLISE DE DADOS E MACHINE LEARNING
# CALCULOS USANDO OS DADOS DA TABELA.

def calcular_totais_e_proporcoes(df):
    """
    Calcula o total de matrículas por tipo de ensino e as proporções.
    """
    # Adicionar colunas de total
    df['Total_Infantil'] = df['Creche_Parcial'] + df['Creche_Integral'] + df['Pre_escola_Parcial'] + df['Pre_escola_Integral']
    df['Total_Fundamental'] = df['Anos_Iniciais_Parcial'] + df['Anos_Iniciais_Integral'] + df['Anos_Finais_Parcial'] + df['Anos_Finais_Integral']
    df['Total_Medio'] = df['Medio_Parcial'] + df['Medio_Integral']
    df['Total_EJA'] = df['EJA_Fundamental'] + df['EJA_Medio']
    df['Total_Geral'] = df['Total_Infantil'] + df['Total_Fundamental'] + df['Total_Medio'] + df['Total_EJA']

    # Calcular as proporções em relação ao total
    df['Prop_Infantil'] = (df['Total_Infantil'] / df['Total_Geral']) * 100
    df['Prop_Fundamental'] = (df['Total_Fundamental'] / df['Total_Geral']) * 100
    df['Prop_Medio'] = (df['Total_Medio'] / df['Total_Geral']) * 100
    df['Prop_EJA'] = (df['Total_EJA'] / df['Total_Geral']) * 100

    return df

def agrupar_por_localizacao(df):
    """
    Agrupa os dados por tipo de localização (Urbana e Rural) e calcula a média.
    """
    print("\n📊 Agrupando dados por localização...")
    
    # Criar uma coluna 'Tipo_Localizacao'
    df['Tipo_Localizacao'] = df['Localizacao'].apply(
        lambda x: 'Urbana' if 'Urbana' in x else 'Rural' if 'Rural' in x else 'Outro'
    )
    
    # Excluir linhas que não são nem 'Urbana' nem 'Rural'
    df = df[df['Tipo_Localizacao'] != 'Outro']
    
    # Agrupar e calcular a média das matrículas
    grouped_df = df.groupby('Tipo_Localizacao')[['Total_Infantil', 'Total_Fundamental', 'Total_Medio', 'Total_EJA']].mean().reset_index()
    
    return grouped_df

def clustering_por_matriula(df): #A matrícula, por ser um identificador único, não pode ser utilizada como critério para agrupar dados semelhantes. 
                                 # Em vez disso, ela é usada para identificar os registros individuais.
    """
    Usa K-Means para agrupar as cidades/regiões com base no perfil de matrículas.
    """
    print("🧠 Aplicando Clustering K-Means...")
    
    # Usar as proporções como features para o clustering
    features = ['Prop_Infantil', 'Prop_Fundamental', 'Prop_Medio', 'Prop_EJA']
    
    # Remover linhas com valores nulos nas features
    df_clustering = df.dropna(subset=features)
    
    X = df_clustering[features]

    # Aplicar K-Means com 3 clusters
    try:
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        df_clustering['Cluster'] = kmeans.fit_predict(X)
        print("✅ Clustering concluído.")
        return df_clustering
    except Exception as e:
        print(f"⚠️ Erro ao aplicar K-Means: {e}")
        return df_clustering


def gerar_graficos(df_localizacao, df_clustering):
    """
    Gera gráficos de barras e de pizza para visualização dos dados.
    """
    print("🎨 Gerando gráficos...")
    # Gráfico 1: Média de Matrículas por Localização (Urbana vs. Rural)
    df_localizacao.set_index('Tipo_Localizacao').plot(kind='bar', figsize=(8, 5))
    plt.title('Média de Matrículas por Localização (Urbana vs. Rural)', fontsize=14)
    plt.xlabel('Localização', fontsize=12)
    plt.ylabel('Média de Matrículas', fontsize=12)
    plt.xticks(rotation=0)
    plt.tight_layout()
    bar_path = os.path.join(OUTPUT_DIR, "matrículas_por_localizacao.png")
    plt.savefig(bar_path)
    plt.close()

    # Gráfico 2: Distribuição de matrículas por cluster
    cluster_counts = df_clustering['Cluster'].value_counts(normalize=True) * 100
    cluster_counts.plot(kind='pie', autopct='%1.1f%%', figsize=(6, 6))
    plt.title('Distribuição de Regiões por Perfil de Matrícula (Clusters)', fontsize=14)
    plt.ylabel('')
    plt.tight_layout()
    pie_path = os.path.join(OUTPUT_DIR, "distribuicao_clusters.png")
    plt.savefig(pie_path)
    plt.close()

    print("✅ Gráficos gerados com sucesso.")
    return bar_path, pie_path

def gerar_pdf(df_localizacao, bar_path, pie_path):
    """
    Cria um relatório em PDF com as análises e gráficos.
    """
    print("\n📄 Gerando relatório em PDF...")
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(PDF_PATH, pagesize=A4)
    story = []

    story.append(Paragraph("<b>Relatório de Análise do Censo Escolar</b>", styles['Title']))
    story.append(Spacer(1, 12))

    intro_text = """
    Este relatório foi gerado automaticamente a partir dos dados do Censo Escolar.
    Ele apresenta uma análise das matrículas por nível de ensino e localização (Urbana/Rural),
    além de agrupar regiões com perfis de matrícula semelhantes usando Machine Learning (K-Means).
    """
    story.append(Paragraph(intro_text, styles['Normal']))
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>1. Média de Matrículas por Localização</b>", styles['Heading2']))
    story.append(Paragraph(
        "Este gráfico compara a média de matrículas em diferentes níveis de ensino nas áreas urbanas e rurais.", 
        styles['Normal']
    ))
    story.append(Image(bar_path, width=450, height=300))
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>2. Agrupamento de Regiões por Perfil de Matrícula</b>", styles['Heading2']))
    story.append(Paragraph(
        "A análise de clustering (K-Means) agrupa as regiões em perfis distintos. "
        "O gráfico de pizza abaixo mostra a porcentagem de regiões em cada um dos três clusters identificados.", 
        styles['Normal']
    ))
    story.append(Image(pie_path, width=400, height=400))
    story.append(Spacer(1, 12))
    
    story.append(Paragraph("<b>3. Principais Métricas</b>", styles['Heading2']))
    df_metrics = df_localizacao.set_index('Tipo_Localizacao').transpose()
    df_metrics.index.name = 'Nível de Ensino'
    df_metrics.reset_index(inplace=True)

    table_data = [list(df_metrics.columns)] + df_metrics.values.tolist()
    table = Table(table_data, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
        ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
    ]))
    story.append(table)
    story.append(Spacer(1, 12))

    doc.build(story)
    print(f"✅ Relatório PDF gerado em: {PDF_PATH}")

# ============================
# PROGRAMA PRINCIPAL
# ============================
if __name__ == "__main__":
    df = carregar_e_limpar_dados()
    df = calcular_totais_e_proporcoes(df)
    
    # Análise de agrupamento
    df_clustering = clustering_por_matriula(df.copy())
    
    # Análise por localização
    df_localizacao = agrupar_por_localizacao(df.copy())
    
    # Geração dos gráficos
    bar_path, pie_path = gerar_graficos(df_localizacao, df_clustering)
    
    # Geração do PDF

    gerar_pdf(df_localizacao, bar_path, pie_path)
