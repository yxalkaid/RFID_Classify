import hydra
from omegaconf import DictConfig
import logging
import time
import torch

import socket
import threading

from model.ClassifyNet.CNNClassifyNet import CNNClassifyNet as ClassifyNet
from model.ModelWorker.ClassifyModelWorker import ClassifyModelWorker
from utils.DataWindow.TimeSeriesWindow import TimeSeriesWindow

from utils.proto import data_pb2

# utils/websocket_server.py
import asyncio
import websockets
import json
from threading import Thread


class WebSocketServer:
    def __init__(self, host="0.0.0.0", port=8765):
        self.host = host
        self.port = port
        self.clients = set()
        self.server = None
        self.thread = None

    async def _handler(self, websocket):
        print("Client connected")
        self.clients.add(websocket)
        try:
            async for message in websocket:
                pass  # 可以处理客户端消息
        finally:
            self.clients.remove(websocket)

    async def _start_server(self):
        self.server = await websockets.serve(self._handler, self.host, self.port)
        print(f"WebSocket server started on ws://{self.host}:{self.port}")
        await self.server.wait_closed()

    def start(self):
        def run():
            asyncio.run(self._start_server())

        self.thread = Thread(target=run)
        self.thread.start()

    def stop(self):
        if self.server:
            self.server.close()
            self.server = None
        if self.thread and self.thread.is_alive():
            self.thread.join()

    async def _send_to_all(self, message):
        if not self.clients:
            return
        disconnected = set()
        for client in self.clients:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(client)
        self.clients -= disconnected

    def send(self, data):
        asyncio.run(self._send_to_all(json.dumps(data)))


class UDPListener:
    def __init__(self, host="0.0.0.0", port=8090):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.host, self.port))
        self.running = True
        self.callback = None

    def start(self, callback):
        """启动UDP监听线程"""
        self.callback = callback
        thread = threading.Thread(target=self.__listen)
        thread.start()

    def __listen(self):
        """监听并处理数据"""

        logging.info("Starting UDP listener...")
        while self.running:
            data, addr = self.sock.recvfrom(1024)  # 缓冲区大小
            print(f"Received {len(data)} bytes from {addr}")
            if self.callback:
                self.callback(data)
        logging.info("Stopped UDP listener.")
        self.sock.close()

    def stop(self):
        """停止UDP监听"""
        self.running = False


@hydra.main(config_path="./config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """
    启动函数
    """
    logging.info("Application Launch")

    input_shape = cfg.model.input_shape
    num_classes = cfg.model.num_classes
    weights_path = cfg.model.weights_path
    print(f"{input_shape=}")
    print(f"{num_classes=}")
    print(f"{weights_path=}")

    C, T, X = input_shape

    # 初始化模型和模型工作器
    model = ClassifyNet(input_shape=input_shape, num_classes=num_classes)
    model_worker = ClassifyModelWorker(model)
    model_worker.load(weights_path)

    data_window = TimeSeriesWindow(T=T, sample_shape=(C, X))

    # 初始化WebSocket服务
    ws_server = WebSocketServer(host="0.0.0.0", port=8765)
    ws_server.start()

    # 初始化UDP监听
    udp_listener = UDPListener(host="0.0.0.0", port=8090)

    def handle_udp_data(data):
        pb_data = data_pb2.DataPoint()
        pb_data.ParseFromString(data)

        print(f"Received data: index={pb_data.index}, time={pb_data.time}")

        # 接收延迟
        delay_ms = int(time.time() * 1000) - int(pb_data.time)
        print(f"UDP packet delay: {delay_ms} ms")

        # 解析 data 字段（二维数组）
        raw_data = list(pb_data.data)
        data_window.add_sample(raw_data, copy=False)
        sample_data = data_window.get_window_data()
        if sample_data is not None:
            inputs = torch.tensor(sample_data, dtype=torch.float32).unsqueeze(0)
            preds, _ = model_worker.predict(inputs)
            print(f"Prediction: {preds}")

            # 推理延迟
            inference_delay_ms = int(time.time() * 1000) - int(pb_data.time)
            print(f"Inference delay: {inference_delay_ms} ms")

            # 通过WebSocket发送结果
            result = {
                "index": pb_data.index,
                "time": pb_data.time,
                "prediction": preds.cpu().tolist(),
                "inference_delay_ms": inference_delay_ms,
            }
            ws_server.send(result)
        print("========================")

    try:
        udp_listener.start(callback=handle_udp_data)
        time.sleep(60)
    except KeyboardInterrupt:
        logging.info("Stopped by user.")
    finally:
        udp_listener.stop()
        ws_server.stop()
        logging.info("Application Exit")


if __name__ == "__main__":
    main()
