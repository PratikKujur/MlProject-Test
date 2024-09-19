import matplotlib.pyplot as plt
import streamlit as st

class ModelEvaluation:
    def __init__(self, results, names):
        self.results = results
        self.names = names

    def evaluate(self):
        fig = plt.figure()
        fig.suptitle('Algorithm Comparison')
        ax = fig.add_subplot(111)
        plt.boxplot(self.results)
        ax.set_xticklabels(self.names)
        st.pyplot(fig)