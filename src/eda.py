import matplotlib.pyplot as plt
import seaborn as sns

def plot_sales_during_holidays(train):
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='StateHoliday', y='Sales', data=train)
    plt.title('Sales During State Holidays')
    plt.show()

def plot_sales_vs_promo(train):
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Promo', y='Sales', data=train)
    plt.title('Sales vs Promo')
    plt.show()
