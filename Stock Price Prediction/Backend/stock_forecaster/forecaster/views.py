from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.decorators import action
from .serializers import StockDataSerializer, PredictionSerializer
from .models import StockPrediction
from .ml_model import StockPredictor
from django.utils import timezone


class StockPredictorViewSet(viewsets.ViewSet):
    predictor = StockPredictor()

    @action(detail=False, methods=['post'])
    def predict(self, request):
        serializer = StockDataSerializer(data=request.data)
        if serializer.is_valid():
            data = serializer.validated_data

            try:
                prediction = self.predictor.predict(data)

                # Save prediction to database
                stock_prediction = StockPrediction.objects.create(
                    symbol=data['symbol'],
                    actual_price=data['last_value'],
                    predicted_price=prediction
                )

                response_data = {
                    'symbol': data['symbol'],
                    'current_price': data['last_value'],
                    'predicted_price': prediction,
                    'timestamp': timezone.now()
                }

                return Response(response_data, status=status.HTTP_200_OK)

            except Exception as e:
                return Response(
                    {'error': str(e)},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    @action(detail=False, methods=['get'])
    def history(self, request):
        predictions = StockPrediction.objects.all().order_by('-date')[:100]
        serializer = PredictionSerializer(predictions, many=True)
        return Response(serializer.data)