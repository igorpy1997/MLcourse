import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.mixture import GaussianMixture
import hdbscan

def cluster_analysis(cleaned_data):
    range_n_clusters = range(2, 11)  # Определяем диапазон кластеров

    # Инициализация словарей для хранения метрик
    metrics = {
        'KMeans': {'inertia': [], 'silhouette': [], 'davies_bouldin': [], 'aic': [], 'bic': []},
        'Agglomerative': {'silhouette': [], 'davies_bouldin': []},
        'HDBSCAN': {'silhouette': [], 'davies_bouldin': []}
    }

    # KMeans и AgglomerativeClustering
    for n_clusters in range_n_clusters:
        # KMeans кластеризация
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(cleaned_data)
        labels_kmeans = kmeans.labels_

        # Инерция (метод локтя)
        metrics['KMeans']['inertia'].append(kmeans.inertia_)

        # Силуэтный коэффициент
        silhouette_avg_kmeans = silhouette_score(cleaned_data, labels_kmeans)
        metrics['KMeans']['silhouette'].append(silhouette_avg_kmeans)

        # Индекс Дэвиса-Болдина
        davies_bouldin_avg_kmeans = davies_bouldin_score(cleaned_data, labels_kmeans)
        metrics['KMeans']['davies_bouldin'].append(davies_bouldin_avg_kmeans)

        # Gaussian Mixture Model (GMM) для AIC и BIC
        gmm = GaussianMixture(n_components=n_clusters, random_state=42).fit(cleaned_data)
        metrics['KMeans']['aic'].append(gmm.aic(cleaned_data))
        metrics['KMeans']['bic'].append(gmm.bic(cleaned_data))

        # Agglomerative кластеризация
        agglomerative = AgglomerativeClustering(n_clusters=n_clusters).fit(cleaned_data)
        labels_agglomerative = agglomerative.labels_

        # Силуэтный коэффициент
        silhouette_avg_agglo = silhouette_score(cleaned_data, labels_agglomerative)
        metrics['Agglomerative']['silhouette'].append(silhouette_avg_agglo)

        # Индекс Дэвиса-Болдина
        davies_bouldin_avg_agglo = davies_bouldin_score(cleaned_data, labels_agglomerative)
        metrics['Agglomerative']['davies_bouldin'].append(davies_bouldin_avg_agglo)

    # HDBSCAN кластеризация
    hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=5)
    labels_hdbscan = hdbscan_model.fit_predict(cleaned_data)

    # Исключаем шум (кластеры с меткой -1)
    cleaned_data_hdbscan = cleaned_data[labels_hdbscan != -1]
    labels_hdbscan = labels_hdbscan[labels_hdbscan != -1]

    # Силуэтный коэффициент
    silhouette_avg_hdbscan = silhouette_score(cleaned_data_hdbscan, labels_hdbscan)
    metrics['HDBSCAN']['silhouette'].append(silhouette_avg_hdbscan)

    # Индекс Дэвиса-Болдина
    davies_bouldin_avg_hdbscan = davies_bouldin_score(cleaned_data_hdbscan, labels_hdbscan)
    metrics['HDBSCAN']['davies_bouldin'].append(davies_bouldin_avg_hdbscan)

    # Построение графиков для каждой метрики
    plt.figure(figsize=(18, 10))

    # Метод локтя (инерция) для KMeans
    plt.subplot(2, 3, 1)
    plt.plot(range_n_clusters, metrics['KMeans']['inertia'], marker='o')
    plt.title('Метод локтя (Inertia) для KMeans')
    plt.xlabel('Количество кластеров')
    plt.ylabel('Inertia')

    # Силуэтный анализ для всех методов
    plt.subplot(2, 3, 2)
    plt.plot(range_n_clusters, metrics['KMeans']['silhouette'], marker='o', label='KMeans')
    plt.plot(range_n_clusters, metrics['Agglomerative']['silhouette'], marker='x', label='Agglomerative')
    plt.scatter([len(metrics['HDBSCAN']['silhouette'])], metrics['HDBSCAN']['silhouette'], marker='*', color='red', label='HDBSCAN')
    plt.title('Силуэтный анализ')
    plt.xlabel('Количество кластеров')
    plt.ylabel('Silhouette Coefficient')
    plt.legend()

    # Индекс Дэвиса-Болдина для всех методов
    plt.subplot(2, 3, 3)
    plt.plot(range_n_clusters, metrics['KMeans']['davies_bouldin'], marker='o', label='KMeans')
    plt.plot(range_n_clusters, metrics['Agglomerative']['davies_bouldin'], marker='x', label='Agglomerative')
    plt.scatter([len(metrics['HDBSCAN']['davies_bouldin'])], metrics['HDBSCAN']['davies_bouldin'], marker='*', color='red', label='HDBSCAN')
    plt.title('Индекс Дэвиса-Болдина')
    plt.xlabel('Количество кластеров')
    plt.ylabel('Davies-Bouldin Index')
    plt.legend()

    # AIC для GMM
    plt.subplot(2, 3, 4)
    plt.plot(range_n_clusters, metrics['KMeans']['aic'], marker='o')
    plt.title('AIC для GMM')
    plt.xlabel('Количество кластеров')
    plt.ylabel('AIC')

    # BIC для GMM
    plt.subplot(2, 3, 5)
    plt.plot(range_n_clusters, metrics['KMeans']['bic'], marker='o')
    plt.title('BIC для GMM')
    plt.xlabel('Количество кластеров')
    plt.ylabel('BIC')

    plt.tight_layout()
    plt.show()

    # Отчет по метрикам
    optimal_clusters = {
        'KMeans Метод локтя': range_n_clusters[np.argmin(metrics['KMeans']['inertia'])],
        'KMeans Силуэтный анализ': range_n_clusters[np.argmax(metrics['KMeans']['silhouette'])],
        'KMeans Индекс Дэвиса-Болдина': range_n_clusters[np.argmin(metrics['KMeans']['davies_bouldin'])],
        'KMeans AIC для GMM': range_n_clusters[np.argmin(metrics['KMeans']['aic'])],
        'KMeans BIC для GMM': range_n_clusters[np.argmin(metrics['KMeans']['bic'])],
        'Agglomerative Силуэтный анализ': range_n_clusters[np.argmax(metrics['Agglomerative']['silhouette'])],
        'Agglomerative Индекс Дэвиса-Болдина': range_n_clusters[np.argmin(metrics['Agglomerative']['davies_bouldin'])],
        'HDBSCAN Силуэтный анализ': len(metrics['HDBSCAN']['silhouette']),
        'HDBSCAN Индекс Дэвиса-Болдина': len(metrics['HDBSCAN']['davies_bouldin'])
    }

    return optimal_clusters
