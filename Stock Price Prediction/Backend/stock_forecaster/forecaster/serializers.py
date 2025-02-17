from rest_framework import serializers
from .models import StockPrediction

class StockDataSerializer(serializers.Serializer):
    open_value = serializers.FloatField()
    high_value = serializers.FloatField()
    low_value = serializers.FloatField()
    turnover = serializers.FloatField()
    change_prev_close_percentage = serializers.FloatField()
    last_value = serializers.FloatField()
    symbol = serializers.CharField(max_length=10)

class PredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = StockPrediction
        fields = '__all__'