from django.db import models


class StockPrediction(models.Model):
    date = models.DateTimeField(auto_now_add=True)
    symbol = models.CharField(max_length=10)
    actual_price = models.FloatField()
    predicted_price = models.FloatField()

    def __str__(self):
        return f"{self.symbol} - {self.date}"