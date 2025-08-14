
import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    """
    LSTM 기반의 시계열 분류 모델.
    Feature Engineering된 데이터 시퀀스를 입력받아 '집중'/'비집중'을 분류합니다.
    
    Args:
        input_dim (int): 입력 특징의 차원 (예: 10)
        hidden_dim (int): LSTM의 은닉 상태 차원
        output_dim (int): 출력 차원 (분류할 클래스 개수)
        num_layers (int): LSTM 레이어의 수
        dropout (float): 드롭아웃 확률
    """
    def __init__(self, input_dim, hidden_dim, output_dim=2, num_layers=2, dropout=0.2):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM layer
        # batch_first=True -> 입력 텐서의 차원을 (batch, seq, feature)로 설정
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Fully connected layer (LSTM의 마지막 출력을 받아 클래스로 분류)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 초기 hidden state와 cell state를 0으로 초기화
        # h0, c0: (num_layers, batch_size, hidden_dim)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # LSTM forward pass
        # out: (batch_size, seq_len, hidden_dim)
        out, _ = self.lstm(x, (h0, c0))

        # 마지막 타임스텝의 출력만 사용 (Many-to-One)
        # out: (batch_size, hidden_dim)
        last_step_out = out[:, -1, :]
        
        # 최종 분류
        final_out = self.fc(last_step_out)
        return final_out
