# utils/training_utils.py
import torch
import gc
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .test_utils import *
import copy
from collections import OrderedDict
from .sam import *

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def train(model, train_loader, val_loader, optimizer, num_epochs, patience, device):
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=2, verbose=True)
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images, labels=labels)
            logits = outputs[1]
            loss = outputs[0].sum()
            
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    # del images, labels, outputs, logits, loss
    # torch.cuda.empty_cache()
    # gc.collect()

        avg_train_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total
        print(f'Epoch {epoch}, Training Loss: {avg_train_loss:.4f}, Accuracy: {accuracy:.2f}%')

        val_loss, val_acc = validate(model, val_loader, device)
        print(f'Epoch {epoch}, Validation Loss: {val_loss:.4f}, Val_Accuracy: {val_acc:.2f}%')

        scheduler.step(val_loss)
        early_stopping(val_loss)

        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    torch.cuda.empty_cache()
    gc.collect()

def validate(model, val_loader, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images, labels=labels)
            loss = outputs[0]
            logits = outputs[1]
            
            running_loss += loss.sum().item()

            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # 메모리 해제
    # del images, labels, outputs, logits, loss
    torch.cuda.empty_cache()
    gc.collect()

    avg_loss = running_loss / len(val_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy
    
####################################################################################################################################


def average_weights(client_weights):
    avg_weights = OrderedDict()
    for key in client_weights[0].keys():
        stacked = torch.stack(
            [client_weight[key].cpu().float() for client_weight in client_weights],
            dim=0
        )
        avg_weights[key] = torch.mean(stacked, dim=0)
    return avg_weights

def average_weights_with_ratio(client_weights, ratio):
    avg_weights = OrderedDict()

    total_ratio = sum(ratio)
    normalized_ratio = [r / total_ratio for r in ratio]

    for key in client_weights[0].keys():
        weighted_sum = torch.zeros_like(client_weights[0][key], dtype=torch.float)
        for client_weight, weight_ratio in zip(client_weights, normalized_ratio):
            weighted_sum += client_weight[key].cpu().float() * weight_ratio
        avg_weights[key] = weighted_sum

    return avg_weights


####################################################################################################################################
# Normal Federated Learning #
####################################################################################################################################
def federated_train(global_model, clients, num_rounds, num_epochs, patience, device, root, test_loader, ex_loader_1, ex_loader_2, ex_loader_3):
    for round_num in range(1, num_rounds + 1):
        print(f"\n=== Federated Round {round_num} / {num_rounds} ===")
        client_weights = []

        for client_id, (client_train_loader, client_val_loader) in enumerate(clients):
            print(f"\n[Client {client_id}] 로컬 학습 시작...")
            local_model = copy.deepcopy(global_model)
            local_model.to(device)

            optimizer = torch.optim.Adam(local_model.parameters(), lr=1e-5)
            train(
                model=local_model, 
                train_loader=client_train_loader, 
                val_loader=client_val_loader, 
                optimizer=optimizer, 
                num_epochs=num_epochs, 
                patience=patience, 
                device=device
            )

            if round_num in [1, 2, 5, 10, 20]:
                test(local_model, client_train_loader, device, f"{root}/train_local_{client_id}")
                test(local_model, client_val_loader, device, f"{root}/valid_local_{client_id}")
                test(local_model, test_loader, device, f"{root}/test_local_{client_id}")
                test(local_model, ex_loader_1, device, f"{root}/ex1_local_{client_id}")
                test(local_model, ex_loader_2, device, f"{root}/ex2_local_{client_id}")
                test(local_model, ex_loader_3, device, f"{root}/ex3_local_{client_id}")

                local_model_path = os.path.join(root, f"local_{client_id}.pth")
                torch.save(local_model.state_dict(), local_model_path)
                print(f"Local model for Client {client_id} saved to {local_model_path}")

            local_model.to("cpu")
            client_weights.append(copy.deepcopy(local_model.state_dict()))

            del local_model
            torch.cuda.empty_cache()
            gc.collect()

        global_weights = average_weights(client_weights)
        global_model.load_state_dict(global_weights)

        if round_num in [1, 2, 5, 10, 20]:
            global_model.to(device)
            test(global_model, test_loader, device, f"{root}/global_{round_num}_test_Global")
            test(global_model, ex_loader_1, device, f"{root}/global_{round_num}_ex1_Global")
            test(global_model, ex_loader_2, device, f"{root}/global_{round_num}_ex2_Global")
            test(global_model, ex_loader_3, device, f"{root}/global_{round_num}_ex3_Glboal")
            global_model.to("cpu")


        print(f"=== End of Federated Round {round_num} ===\n")
        torch.cuda.empty_cache()
        gc.collect()

    return global_model

def imp_federated_train(global_model, clients, num_rounds, num_epochs, patience, device, root, test_loader, ex_loader_1, ex_loader_2, ex_loader_3):
    for round_num in range(1, num_rounds + 1):
        print(f"\n=== Federated Round {round_num} / {num_rounds} ===")
        client_weights = []
        accs = []
        for client_id, (client_train_loader, client_val_loader) in enumerate(clients):
            print(f"\n[Client {client_id}] 로컬 학습 시작...")
            local_model = copy.deepcopy(global_model)
            local_model.to(device)

            optimizer = torch.optim.Adam(local_model.parameters(), lr=1e-5)
            train(
                model=local_model, 
                train_loader=client_train_loader, 
                val_loader=client_val_loader, 
                optimizer=optimizer, 
                num_epochs=num_epochs, 
                patience=patience, 
                device=device
            )

            _, acc = validate(local_model, test_loader, device)

            if round_num in [1, 2, 5, 10, 20]:
                test(local_model, client_train_loader, device, f"{root}/global_{round_num}_train_local_{client_id}")
                test(local_model, client_val_loader, device, f"{root}/global_{round_num}_valid_local_{client_id}")
                test(local_model, test_loader, device, f"{root}/global_{round_num}_test_local_{client_id}")
                test(local_model, ex_loader_1, device, f"{root}/global_{round_num}_ex1_local_{client_id}")
                test(local_model, ex_loader_2, device, f"{root}/global_{round_num}_ex2_local_{client_id}")
                test(local_model, ex_loader_3, device, f"{root}/global_{round_num}_ex3_local_{client_id}")

                local_model_path = os.path.join(root, f"local_{client_id}.pth")
                torch.save(local_model.state_dict(), local_model_path)
                print(f"Local model for Client {client_id} saved to {local_model_path}")

            local_model.to("cpu")
            client_weights.append(copy.deepcopy(local_model.state_dict()))
            accs.append(acc)

            del local_model
            torch.cuda.empty_cache()
            gc.collect()

        global_weights = average_weights_with_ratio(client_weights, accs)
        global_model.load_state_dict(global_weights)

        if round_num in [1, 2, 5, 10, 20]:
            global_model.to(device)
            test(global_model, test_loader, device, f"{root}/global_{round_num}_test_Global")
            test(global_model, ex_loader_1, device, f"{root}/global_{round_num}_ex1_Global")
            test(global_model, ex_loader_2, device, f"{root}/global_{round_num}_ex2_Global")
            test(global_model, ex_loader_3, device, f"{root}/global_{round_num}_ex3_Glboal")
            global_model.to("cpu")


        print(f"=== End of Federated Round {round_num} ===\n")
        torch.cuda.empty_cache()
        gc.collect()

    return global_model


def ipa_federated_train(global_model, clients, num_rounds, num_epochs, patience, device, root, test_loader, ex_loader_1, ex_loader_2, ex_loader_3):
    paired_clients = [(clients[i], clients[i+1]) for i in range(0, len(clients), 2)]

    for round_num in range(1, num_rounds + 1):
        print(f"\n=== Federated Round {round_num} / {num_rounds} ===")

        for pair_num, (client_1, client_2) in enumerate(paired_clients):
            client_weights = []
            print(f"\n[Pair {pair_num + 1}] 로컬 학습 시작...")

            local_model_1 = copy.deepcopy(global_model)
            local_model_1.to(device)
            optimizer_1 = torch.optim.Adam(local_model_1.parameters(), lr=1e-5)
            train(
                model=local_model_1, 
                train_loader=client_1[0], 
                val_loader=client_1[1], 
                optimizer=optimizer_1, 
                num_epochs=num_epochs, 
                patience=patience, 
                device=device
            )

            if round_num in [1, 2, 5, 10, 20]:
                test(local_model_1, client_1[0], device, f"{root}/global_{round_num}_train_local_{pair_num+1}_client1")
                test(local_model_1, client_1[1], device, f"{root}/global_{round_num}_valid_local_{pair_num+1}_client1")
                test(local_model_1, test_loader, device, f"{root}/global_{round_num}_test_local_{pair_num+1}_client1")
                test(local_model_1, ex_loader_1, device, f"{root}/global_{round_num}_ex1_local_{pair_num+1}_client1")
                test(local_model_1, ex_loader_2, device, f"{root}/global_{round_num}_ex2_local_{pair_num+1}_client1")
                test(local_model_1, ex_loader_3, device, f"{root}/global_{round_num}_ex3_local_{pair_num+1}_client1")

                local_model_path = os.path.join(root, f"{pair_num+1}_client1.pth")
                torch.save(local_model_1.state_dict(), local_model_path)
                print(f"Local model for Client 1 saved to {local_model_path}")
            local_model_1.to("cpu")

            local_model_2 = copy.deepcopy(global_model)
            local_model_2.to(device)
            optimizer_2 = torch.optim.Adam(local_model_2.parameters(), lr=1e-5)
            train(
                model=local_model_2, 
                train_loader=client_2[0], 
                val_loader=client_2[1], 
                optimizer=optimizer_2, 
                num_epochs=num_epochs, 
                patience=patience, 
                device=device
            )

            if round_num in [1, 2, 5, 10, 20]:
                test(local_model_2, client_2[0], device, f"{root}/global_{round_num}_train_local_{pair_num+1}_client2")
                test(local_model_2, client_2[1], device, f"{root}/global_{round_num}_valid_local_{pair_num+1}_client2")
                test(local_model_2, test_loader, device, f"{root}/global_{round_num}_test_local_{pair_num+1}_client2")
                test(local_model_2, ex_loader_1, device, f"{root}/global_{round_num}_ex1_local_{pair_num+1}_client2")
                test(local_model_2, ex_loader_2, device, f"{root}/global_{round_num}_ex2_local_{pair_num+1}_client2")
                test(local_model_2, ex_loader_3, device, f"{root}/global_{round_num}_ex3_local_{pair_num+1}_client2")

                local_model_path = os.path.join(root, f"{pair_num+1}_client2.pth")
                torch.save(local_model_2.state_dict(), local_model_path)
                print(f"Local model for Client 2 saved to {local_model_path}")
            local_model_2.to("cpu")

            client_weights.append(copy.deepcopy(local_model_1.state_dict()))
            client_weights.append(copy.deepcopy(local_model_2.state_dict()))
            global_weights = average_weights(client_weights)
            global_model.load_state_dict(global_weights)

        if round_num in [1, 2, 5, 10, 20]:
            global_model.to(device)
            test(global_model, test_loader, device, f"{root}/global_{round_num}_test_Global")
            test(global_model, ex_loader_1, device, f"{root}/global_{round_num}_ex1_Global")
            test(global_model, ex_loader_2, device, f"{root}/global_{round_num}_ex2_Global")
            test(global_model, ex_loader_3, device, f"{root}/global_{round_num}_ex3_Glboal")
            global_model.to("cpu")


            del local_model_1, local_model_2
            torch.cuda.empty_cache()
            gc.collect()

        print(f"=== End of Federated Round {round_num} ===\n")
        torch.cuda.empty_cache()
        gc.collect()

    return global_model

def imp_ipa_federated_train(global_model, clients, num_rounds, num_epochs, patience, device, root, test_loader, ex_loader_1, ex_loader_2, ex_loader_3):
    paired_clients = [(clients[i], clients[i+1]) for i in range(0, len(clients), 2)]

    for round_num in range(1, num_rounds + 1):
        print(f"\n=== Federated Round {round_num} / {num_rounds} ===")

        for pair_num, (client_1, client_2) in enumerate(paired_clients):
            client_weights = []
            accs = []
            print(f"\n[Pair {pair_num + 1}] 로컬 학습 시작...")

            local_model_1 = copy.deepcopy(global_model)
            local_model_1.to(device)
            optimizer_1 = torch.optim.Adam(local_model_1.parameters(), lr=1e-5)
            train(
                model=local_model_1, 
                train_loader=client_1[0], 
                val_loader=client_1[1], 
                optimizer=optimizer_1, 
                num_epochs=num_epochs, 
                patience=patience, 
                device=device
            )
            _, acc = validate(local_model_1, test_loader, device)
            accs.append(acc)

            if round_num in [1, 2, 5, 10, 20]:
                test(local_model_1, client_1[0], device, f"{root}/global_{round_num}_train_local_{pair_num+1}_client1")
                test(local_model_1, client_1[1], device, f"{root}/global_{round_num}_valid_local_{pair_num+1}_client1")
                test(local_model_1, test_loader, device, f"{root}/global_{round_num}_test_local_{pair_num+1}_client1")
                test(local_model_1, ex_loader_1, device, f"{root}/global_{round_num}_ex1_local_{pair_num+1}_client1")
                test(local_model_1, ex_loader_2, device, f"{root}/global_{round_num}_ex2_local_{pair_num+1}_client1")
                test(local_model_1, ex_loader_3, device, f"{root}/global_{round_num}_ex3_local_{pair_num+1}_client1")

                local_model_path = os.path.join(root, f"{pair_num+1}_client1.pth")
                torch.save(local_model_1.state_dict(), local_model_path)
                print(f"Local model for Client 1 saved to {local_model_path}")
            local_model_1.to("cpu")

            local_model_2 = copy.deepcopy(global_model)
            local_model_2.to(device)
            optimizer_2 = torch.optim.Adam(local_model_2.parameters(), lr=1e-5)
            train(
                model=local_model_2, 
                train_loader=client_2[0], 
                val_loader=client_2[1], 
                optimizer=optimizer_2, 
                num_epochs=num_epochs, 
                patience=patience, 
                device=device
            )
            _, acc = validate(local_model_2, test_loader, device)
            accs.append(acc)

            if round_num in [1, 2, 5, 10, 20]:
                test(local_model_2, client_2[0], device, f"{root}/global_{round_num}_train_local_{pair_num+1}_client2")
                test(local_model_2, client_2[1], device, f"{root}/global_{round_num}_valid_local_{pair_num+1}_client2")
                test(local_model_2, test_loader, device, f"{root}/global_{round_num}_test_local_{pair_num+1}_client2")
                test(local_model_2, ex_loader_1, device, f"{root}/global_{round_num}_ex1_local_{pair_num+1}_client2")
                test(local_model_2, ex_loader_2, device, f"{root}/global_{round_num}_ex2_local_{pair_num+1}_client2")
                test(local_model_2, ex_loader_3, device, f"{root}/global_{round_num}_ex3_local_{pair_num+1}_client2")

                local_model_path = os.path.join(root, f"{pair_num+1}_client2.pth")
                torch.save(local_model_2.state_dict(), local_model_path)
                print(f"Local model for Client 2 saved to {local_model_path}")
            local_model_2.to("cpu")

            client_weights.append(copy.deepcopy(local_model_1.state_dict()))
            client_weights.append(copy.deepcopy(local_model_2.state_dict()))
            global_weights = average_weights_with_ratio(client_weights, accs)
            global_model.load_state_dict(global_weights)

        if round_num in [1, 2, 5, 10, 20]:
            global_model.to(device)
            test(global_model, test_loader, device, f"{root}/global_{round_num}_test_Global")
            test(global_model, ex_loader_1, device, f"{root}/global_{round_num}_ex1_Global")
            test(global_model, ex_loader_2, device, f"{root}/global_{round_num}_ex2_Global")
            test(global_model, ex_loader_3, device, f"{root}/global_{round_num}_ex3_Glboal")
            global_model.to("cpu")


            del local_model_1, local_model_2
            torch.cuda.empty_cache()
            gc.collect()

        print(f"=== End of Federated Round {round_num} ===\n")
        torch.cuda.empty_cache()
        gc.collect()

    return global_model

####################################################################################################################################
# Feedback #
####################################################################################################################################
def federated_train_feedback(global_model, clients, num_rounds, num_epochs, patience, device, root, test_loader, ex_loader_1, ex_loader_2, ex_loader_3):
    global_model.to('cpu')
    weights_for_feed = copy.deepcopy(global_model.state_dict())

    for round_num in range(1, num_rounds + 1):
        print(f"\n=== Federated Round {round_num} / {num_rounds} ===")
        client_weights = []

        for client_id, (client_train_loader, client_val_loader) in enumerate(clients):
            print(f"\n[Client {client_id}] 로컬 학습 시작...")
            local_model = copy.deepcopy(global_model)
            local_model.to(device)

            optimizer = torch.optim.Adam(local_model.parameters(), lr=1e-5)
            train(
                model=local_model, 
                train_loader=client_train_loader, 
                val_loader=client_val_loader, 
                optimizer=optimizer, 
                num_epochs=num_epochs, 
                patience=patience, 
                device=device
            )

            if round_num in [2, 5, 10, 20]:
                test(local_model, client_train_loader, device, f"{root}/global_{round_num}_train_local_{client_id}")
                test(local_model, client_val_loader, device, f"{root}/global_{round_num}_valid_local_{client_id}")
                test(local_model, test_loader, device, f"{root}/global_{round_num}_test_local_{client_id}")
                test(local_model, ex_loader_1, device, f"{root}/global_{round_num}_ex1_local_{client_id}")
                test(local_model, ex_loader_2, device, f"{root}/global_{round_num}_ex2_local_{client_id}")
                test(local_model, ex_loader_3, device, f"{root}/global_{round_num}_ex3_local_{client_id}")

                local_model_path = os.path.join(root, f"local_{client_id}.pth")
                torch.save(local_model.state_dict(), local_model_path)
                print(f"Local model for Client {client_id} saved to {local_model_path}")
            local_model.to("cpu")
            client_weights.append(copy.deepcopy(local_model.state_dict()))

            del local_model
            torch.cuda.empty_cache()
            gc.collect()

        local_avg_weights = average_weights(client_weights)
        global_weights = average_weights([local_avg_weights, weights_for_feed])
        global_model.load_state_dict(global_weights)
        weights_for_feed = global_weights

        if round_num in [2, 5, 10, 20]:
            global_model.to(device)
            test(global_model, test_loader, device, f"{root}/global_{round_num}_test_Global")
            test(global_model, ex_loader_1, device, f"{root}/global_{round_num}_ex1_Global")
            test(global_model, ex_loader_2, device, f"{root}/global_{round_num}_ex2_Global")
            test(global_model, ex_loader_3, device, f"{root}/global_{round_num}_ex3_Glboal")
            global_model.to("cpu")

            
        print(f"=== End of Federated Round {round_num} ===\n")
        torch.cuda.empty_cache()
        gc.collect()

    return global_model

def imp_federated_train_feedback(global_model, clients, num_rounds, num_epochs, patience, device, root, test_loader, ex_loader_1, ex_loader_2, ex_loader_3):
    global_model.to('cpu')
    weights_for_feed = copy.deepcopy(global_model.state_dict())

    for round_num in range(1, num_rounds + 1):
        print(f"\n=== Federated Round {round_num} / {num_rounds} ===")
        client_weights = []
        accs = []
        for client_id, (client_train_loader, client_val_loader) in enumerate(clients):
            print(f"\n[Client {client_id}] 로컬 학습 시작...")
            local_model = copy.deepcopy(global_model)
            local_model.to(device)

            optimizer = torch.optim.Adam(local_model.parameters(), lr=1e-5)
            train(
                model=local_model, 
                train_loader=client_train_loader, 
                val_loader=client_val_loader, 
                optimizer=optimizer, 
                num_epochs=num_epochs, 
                patience=patience, 
                device=device
            )

            _, acc = validate(local_model, test_loader, device)

            if round_num in [2, 5, 10, 20]:
                test(local_model, client_train_loader, device, f"{root}/global_{round_num}_train_local_{client_id}")
                test(local_model, client_val_loader, device, f"{root}/global_{round_num}_valid_local_{client_id}")
                test(local_model, test_loader, device, f"{root}/global_{round_num}_test_local_{client_id}")
                test(local_model, ex_loader_1, device, f"{root}/global_{round_num}_ex1_local_{client_id}")
                test(local_model, ex_loader_2, device, f"{root}/global_{round_num}_ex2_local_{client_id}")
                test(local_model, ex_loader_3, device, f"{root}/global_{round_num}_ex3_local_{client_id}")

                local_model_path = os.path.join(root, f"local_{client_id}.pth")
                torch.save(local_model.state_dict(), local_model_path)
                print(f"Local model for Client {client_id} saved to {local_model_path}")
            local_model.to("cpu")
            client_weights.append(copy.deepcopy(local_model.state_dict()))
            accs.append(acc)

            del local_model
            torch.cuda.empty_cache()
            gc.collect()

        local_avg_weights = average_weights_with_ratio(client_weights, accs)
        global_weights = average_weights([local_avg_weights, weights_for_feed])
        global_model.load_state_dict(global_weights)

        if round_num in [2, 5, 10, 20]:
            global_model.to(device)
            test(global_model, test_loader, device, f"{root}/global_{round_num}_test_Global")
            test(global_model, ex_loader_1, device, f"{root}/global_{round_num}_ex1_Global")
            test(global_model, ex_loader_2, device, f"{root}/global_{round_num}_ex2_Global")
            test(global_model, ex_loader_3, device, f"{root}/global_{round_num}_ex3_Glboal")
            global_model.to("cpu")


        weights_for_feed = global_weights

        print(f"=== End of Federated Round {round_num} ===\n")
        torch.cuda.empty_cache()
        gc.collect()

    return global_model

def ipa_federated_train_feedback(global_model, clients, num_rounds, num_epochs, patience, device, root, test_loader, ex_loader_1, ex_loader_2, ex_loader_3):
    global_model.to('cpu')
    paired_clients = [(clients[i], clients[i+1]) for i in range(0, len(clients), 2)]
    weights_for_feed = copy.deepcopy(global_model.state_dict())

    for round_num in range(1, num_rounds + 1):
        print(f"\n=== Federated Round {round_num} / {num_rounds} ===")

        for pair_num, (client_1, client_2) in enumerate(paired_clients):
            client_weights = []
            print(f"\n[Pair {pair_num + 1}] 로컬 학습 시작...")

            local_model_1 = copy.deepcopy(global_model)
            local_model_1.to(device)
            optimizer_1 = torch.optim.Adam(local_model_1.parameters(), lr=1e-5)

            train(
                model=local_model_1, 
                train_loader=client_1[0], 
                val_loader=client_1[1], 
                optimizer=optimizer_1, 
                num_epochs=num_epochs, 
                patience=patience, 
                device=device
            )

            if round_num in [2, 5, 10, 20]:
                test(local_model_1, client_1[0], device, f"{root}/global_{round_num}_train_local_{pair_num+1}_client1")
                test(local_model_1, client_1[1], device, f"{root}/global_{round_num}_valid_local_{pair_num+1}_client1")
                test(local_model_1, test_loader, device, f"{root}/global_{round_num}_test_local_{pair_num+1}_client1")
                test(local_model_1, ex_loader_1, device, f"{root}/global_{round_num}_ex1_local_{pair_num+1}_client1")
                test(local_model_1, ex_loader_2, device, f"{root}/global_{round_num}_ex2_local_{pair_num+1}_client1")
                test(local_model_1, ex_loader_3, device, f"{root}/global_{round_num}_ex3_local_{pair_num+1}_client1")

                local_model_path = os.path.join(root, f"{pair_num+1}_client1.pth")
                torch.save(local_model_1.state_dict(), local_model_path)
                print(f"Local model for Client 1 saved to {local_model_path}")
            local_model_1.to("cpu")


            local_model_2 = copy.deepcopy(global_model)
            local_model_2.to(device)
            optimizer_2 = torch.optim.Adam(local_model_2.parameters(), lr=1e-5)

            train(
                model=local_model_2, 
                train_loader=client_2[0], 
                val_loader=client_2[1], 
                optimizer=optimizer_2, 
                num_epochs=num_epochs, 
                patience=patience, 
                device=device
            )

            if round_num in [2, 5, 10, 20]:
                test(local_model_2, client_2[0], device, f"{root}/global_{round_num}_train_local_{pair_num+1}_client2")
                test(local_model_2, client_2[1], device, f"{root}/global_{round_num}_valid_local_{pair_num+1}_client2")
                test(local_model_2, test_loader, device, f"{root}/global_{round_num}_test_local_{pair_num+1}_client2")
                test(local_model_2, ex_loader_1, device, f"{root}/global_{round_num}_ex1_local_{pair_num+1}_client2")
                test(local_model_2, ex_loader_2, device, f"{root}/global_{round_num}_ex2_local_{pair_num+1}_client2")
                test(local_model_2, ex_loader_3, device, f"{root}/global_{round_num}_ex3_local_{pair_num+1}_client2")

                local_model_path = os.path.join(root, f"{pair_num+1}_client2.pth")
                torch.save(local_model_2.state_dict(), local_model_path)
                print(f"Local model for Client 2 saved to {local_model_path}")

            local_model_2.to("cpu")

            client_weights.append(copy.deepcopy(local_model_1.state_dict()))
            client_weights.append(copy.deepcopy(local_model_2.state_dict()))
            ipa_local_avg_weights = average_weights(client_weights)
            global_model.load_state_dict(ipa_local_avg_weights)

            del local_model_1, local_model_2
            torch.cuda.empty_cache()
            gc.collect()

        global_weights = average_weights([ipa_local_avg_weights, weights_for_feed])
        global_model.load_state_dict(global_weights)
        weights_for_feed = global_weights

        if round_num in [2, 5, 10, 20]:
            global_model.to(device)
            test(global_model, test_loader, device, f"{root}/global_{round_num}_test_Global")
            test(global_model, ex_loader_1, device, f"{root}/global_{round_num}_ex1_Global")
            test(global_model, ex_loader_2, device, f"{root}/global_{round_num}_ex2_Global")
            test(global_model, ex_loader_3, device, f"{root}/global_{round_num}_ex3_Glboal")
            global_model.to("cpu")


        print(f"=== End of Federated Round {round_num} ===\n")
        torch.cuda.empty_cache()
        gc.collect()

    return global_model

def imp_ipa_federated_train_feedback(global_model, clients, num_rounds, num_epochs, patience, device, root, test_loader, ex_loader_1, ex_loader_2, ex_loader_3):
    global_model.to('cpu')
    paired_clients = [(clients[i], clients[i+1]) for i in range(0, len(clients), 2)]
    weights_for_feed = copy.deepcopy(global_model.state_dict())

    for round_num in range(1, num_rounds + 1):
        print(f"\n=== Federated Round {round_num} / {num_rounds} ===")

        for pair_num, (client_1, client_2) in enumerate(paired_clients):
            client_weights = []
            accs = []
            print(f"\n[Pair {pair_num + 1}] 로컬 학습 시작...")

            local_model_1 = copy.deepcopy(global_model)
            local_model_1.to(device)
            optimizer_1 = torch.optim.Adam(local_model_1.parameters(), lr=1e-5)

            train(
                model=local_model_1, 
                train_loader=client_1[0], 
                val_loader=client_1[1], 
                optimizer=optimizer_1, 
                num_epochs=num_epochs, 
                patience=patience, 
                device=device
            )

            _, acc = validate(local_model_1, test_loader, device)
            accs.append(acc)

            if round_num in [2, 5, 10, 20]:
                test(local_model_1, client_1[0], device, f"{root}/global_{round_num}_train_local_{pair_num+1}_client1")
                test(local_model_1, client_1[1], device, f"{root}/global_{round_num}_valid_local_{pair_num+1}_client1")
                test(local_model_1, test_loader, device, f"{root}/global_{round_num}_test_local_{pair_num+1}_client1")
                test(local_model_1, ex_loader_1, device, f"{root}/global_{round_num}_ex1_local_{pair_num+1}_client1")
                test(local_model_1, ex_loader_2, device, f"{root}/global_{round_num}_ex2_local_{pair_num+1}_client1")
                test(local_model_1, ex_loader_3, device, f"{root}/global_{round_num}_ex3_local_{pair_num+1}_client1")

                local_model_path = os.path.join(root, f"{pair_num+1}_client1.pth")
                torch.save(local_model_1.state_dict(), local_model_path)
                print(f"Local model for Client 1 saved to {local_model_path}")
            local_model_1.to("cpu")

            local_model_2 = copy.deepcopy(global_model)
            local_model_2.to(device)
            optimizer_2 = torch.optim.Adam(local_model_2.parameters(), lr=1e-5)


            train(
                model=local_model_2, 
                train_loader=client_2[0], 
                val_loader=client_2[1], 
                optimizer=optimizer_2, 
                num_epochs=num_epochs, 
                patience=patience, 
                device=device
            )

            _, acc = validate(local_model_2, test_loader, device)
            accs.append(acc)

            if round_num in [2, 5, 10, 20]:
                test(local_model_2, client_2[0], device, f"{root}/global_{round_num}_train_local_{pair_num+1}_client2")
                test(local_model_2, client_2[1], device, f"{root}/global_{round_num}_valid_local_{pair_num+1}_client2")
                test(local_model_2, test_loader, device, f"{root}/global_{round_num}_test_local_{pair_num+1}_client2")
                test(local_model_2, ex_loader_1, device, f"{root}/global_{round_num}_ex1_local_{pair_num+1}_client2")
                test(local_model_2, ex_loader_2, device, f"{root}/global_{round_num}_ex2_local_{pair_num+1}_client2")
                test(local_model_2, ex_loader_3, device, f"{root}/global_{round_num}_ex3_local_{pair_num+1}_client2")

                local_model_path = os.path.join(root, f"{pair_num+1}_client2.pth")
                torch.save(local_model_2.state_dict(), local_model_path)
                print(f"Local model for Client 2 saved to {local_model_path}")

            local_model_2.to("cpu")
            
            client_weights.append(copy.deepcopy(local_model_1.state_dict()))
            client_weights.append(copy.deepcopy(local_model_2.state_dict()))
            ipa_local_avg_weights = average_weights_with_ratio(client_weights, accs)
            global_model.load_state_dict(ipa_local_avg_weights)

            del local_model_1, local_model_2
            torch.cuda.empty_cache()
            gc.collect()

        global_weights = average_weights([ipa_local_avg_weights, weights_for_feed])
        global_model.load_state_dict(global_weights)
        weights_for_feed = global_weights

        if round_num in [2, 5, 10, 20]:
            global_model.to(device)
            test(global_model, test_loader, device, f"{root}/global_{round_num}_test_Global")
            test(global_model, ex_loader_1, device, f"{root}/global_{round_num}_ex1_Global")
            test(global_model, ex_loader_2, device, f"{root}/global_{round_num}_ex2_Global")
            test(global_model, ex_loader_3, device, f"{root}/global_{round_num}_ex3_Glboal")
            global_model.to("cpu")


        print(f"=== End of Federated Round {round_num} ===\n")
        torch.cuda.empty_cache()
        gc.collect()

    return global_model

####################################################################################################################################
# Feedback_ratio_performance #
####################################################################################################################################
def federated_train_feedback_performance(global_model, clients, num_rounds, num_epochs, patience, device, root, test_loader, ex_loader_1, ex_loader_2, ex_loader_3):
    global_model.to('cpu')
    weights_for_feed = copy.deepcopy(global_model.state_dict())
    global_model.to(device)
    _, acc1 = validate(global_model, test_loader, device)
    global_model.to('cpu')
    
    for round_num in range(1, num_rounds + 1):
        print(f"\n=== Federated Round {round_num} / {num_rounds} ===")
        client_weights = []

        for client_id, (client_train_loader, client_val_loader) in enumerate(clients):
            print(f"\n[Client {client_id}] 로컬 학습 시작...")
            local_model = copy.deepcopy(global_model)
            local_model.to(device)

            optimizer = torch.optim.Adam(local_model.parameters(), lr=1e-5)
            train(
                model=local_model, 
                train_loader=client_train_loader, 
                val_loader=client_val_loader, 
                optimizer=optimizer, 
                num_epochs=num_epochs, 
                patience=patience, 
                device=device
            )

            if round_num in [2, 5, 10, 20]:
                test(local_model, client_train_loader, device, f"{root}/global_{round_num}_train_local_{client_id}")
                test(local_model, client_val_loader, device, f"{root}/global_{round_num}_valid_local_{client_id}")
                test(local_model, test_loader, device, f"{root}/global_{round_num}_test_local_{client_id}")
                test(local_model, ex_loader_1, device, f"{root}/global_{round_num}_ex1_local_{client_id}")
                test(local_model, ex_loader_2, device, f"{root}/global_{round_num}_ex2_local_{client_id}")
                test(local_model, ex_loader_3, device, f"{root}/global_{round_num}_ex3_local_{client_id}")

                local_model_path = os.path.join(root, f"local_{client_id}.pth")
                torch.save(local_model.state_dict(), local_model_path)
                print(f"Local model for Client {client_id} saved to {local_model_path}")

            local_model.to("cpu")
            client_weights.append(copy.deepcopy(local_model.state_dict()))

            del local_model
            torch.cuda.empty_cache()
            gc.collect()

        local_avg_weights = average_weights(client_weights)
        global_model.load_state_dict(local_avg_weights)
        global_model.to(device)
        _, acc2 = validate(global_model, test_loader, device)
        global_model.to('cpu')
        torch.cuda.empty_cache()
        gc.collect()

        global_weights = average_weights_with_ratio([weights_for_feed, local_avg_weights], [acc1, acc2])
        global_model.load_state_dict(global_weights)

        if round_num in [2, 5, 10, 20]:
            global_model.to(device)
            test(global_model, test_loader, device, f"{root}/global_{round_num}_test_Global")
            test(global_model, ex_loader_1, device, f"{root}/global_{round_num}_ex1_Global")
            test(global_model, ex_loader_2, device, f"{root}/global_{round_num}_ex2_Global")
            test(global_model, ex_loader_3, device, f"{root}/global_{round_num}_ex3_Glboal")
            global_model.to("cpu")

        weights_for_feed = global_weights
        acc1 = acc2

        print(f"=== End of Federated Round {round_num} ===\n")
        torch.cuda.empty_cache()
        gc.collect()

    return global_model

def imp_federated_train_feedback_performance(global_model, clients, num_rounds, num_epochs, patience, device, root, test_loader, ex_loader_1, ex_loader_2, ex_loader_3):
    global_model.to('cpu')
    weights_for_feed = copy.deepcopy(global_model.state_dict())
    global_model.to(device)
    _, acc1 = validate(global_model, test_loader, device)
    global_model.to('cpu')

    for round_num in range(1, num_rounds + 1):
        print(f"\n=== Federated Round {round_num} / {num_rounds} ===")
        client_weights = []
        accs = []
        for client_id, (client_train_loader, client_val_loader) in enumerate(clients):
            print(f"\n[Client {client_id}] 로컬 학습 시작...")
            local_model = copy.deepcopy(global_model)
            local_model.to(device)

            optimizer = torch.optim.Adam(local_model.parameters(), lr=1e-5)
            train(
                model=local_model, 
                train_loader=client_train_loader, 
                val_loader=client_val_loader, 
                optimizer=optimizer, 
                num_epochs=num_epochs, 
                patience=patience, 
                device=device
            )

            _, acc = validate(local_model, test_loader, device)

            if round_num in [2, 5, 10, 20]:
                test(local_model, client_train_loader, device, f"{root}/global_{round_num}_train_local_{client_id}")
                test(local_model, client_val_loader, device, f"{root}/global_{round_num}_valid_local_{client_id}")
                test(local_model, test_loader, device, f"{root}/global_{round_num}_test_local_{client_id}")
                test(local_model, ex_loader_1, device, f"{root}/global_{round_num}_ex1_local_{client_id}")
                test(local_model, ex_loader_2, device, f"{root}/global_{round_num}_ex2_local_{client_id}")
                test(local_model, ex_loader_3, device, f"{root}/global_{round_num}_ex3_local_{client_id}")

                local_model_path = os.path.join(root, f"local_{client_id}.pth")
                torch.save(local_model.state_dict(), local_model_path)
                print(f"Local model for Client {client_id} saved to {local_model_path}")

            local_model.to("cpu")
            torch.cuda.empty_cache()
            gc.collect()
            client_weights.append(copy.deepcopy(local_model.state_dict()))
            accs.append(acc)

            del local_model
            torch.cuda.empty_cache()
            gc.collect()

        local_avg_weights = average_weights_with_ratio(client_weights, accs)
        global_model.load_state_dict(local_avg_weights)
        global_model.to(device)
        _, acc2 = validate(global_model, test_loader, device)
        global_model.to('cpu')
        torch.cuda.empty_cache()
        gc.collect()

        global_weights = average_weights_with_ratio([weights_for_feed, local_avg_weights], [acc1, acc2])
        global_model.load_state_dict(global_weights)

        if round_num in [2, 5, 10, 20]:
            global_model.to(device)
            test(global_model, test_loader, device, f"{root}/global_{round_num}_test_Global")
            test(global_model, ex_loader_1, device, f"{root}/global_{round_num}_ex1_Global")
            test(global_model, ex_loader_2, device, f"{root}/global_{round_num}_ex2_Global")
            test(global_model, ex_loader_3, device, f"{root}/global_{round_num}_ex3_Glboal")
            global_model.to("cpu")


        weights_for_feed = global_weights
        acc1 = acc2

        print(f"=== End of Federated Round {round_num} ===\n")
        torch.cuda.empty_cache()
        gc.collect()

    return global_model

def ipa_federated_train_feedback_performance(global_model, clients, num_rounds, num_epochs, patience, device, root, test_loader, ex_loader_1, ex_loader_2, ex_loader_3):
    global_model.to('cpu')
    paired_clients = [(clients[i], clients[i+1]) for i in range(0, len(clients), 2)]
    weights_for_feed = copy.deepcopy(global_model.state_dict())
    global_model.to(device)
    _, acc1 = validate(global_model, test_loader, device)
    global_model.to('cpu')

    for round_num in range(1, num_rounds + 1):
        print(f"\n=== Federated Round {round_num} / {num_rounds} ===")

        for pair_num, (client_1, client_2) in enumerate(paired_clients):
            client_weights = []
            print(f"\n[Pair {pair_num + 1}] 로컬 학습 시작...")

            local_model_1 = copy.deepcopy(global_model)
            optimizer_1 = torch.optim.Adam(local_model_1.parameters(), lr=1e-5)
            local_model_1.to(device)

            train(
                model=local_model_1, 
                train_loader=client_1[0], 
                val_loader=client_1[1], 
                optimizer=optimizer_1, 
                num_epochs=num_epochs, 
                patience=patience, 
                device=device
            )

            if round_num in [2, 5, 10, 20]:
                test(local_model_1, client_1[0], device, f"{root}/global_{round_num}_train_local_{pair_num+1}_client1")
                test(local_model_1, client_1[1], device, f"{root}/global_{round_num}_valid_local_{pair_num+1}_client1")
                test(local_model_1, test_loader, device, f"{root}/global_{round_num}_test_local_{pair_num+1}_client1")
                test(local_model_1, ex_loader_1, device, f"{root}/global_{round_num}_ex1_local_{pair_num+1}_client1")
                test(local_model_1, ex_loader_2, device, f"{root}/global_{round_num}_ex2_local_{pair_num+1}_client1")
                test(local_model_1, ex_loader_3, device, f"{root}/global_{round_num}_ex3_local_{pair_num+1}_client1")

                local_model_path = os.path.join(root, f"{pair_num+1}_client1.pth")
                torch.save(local_model_1.state_dict(), local_model_path)
                print(f"Local model for Client 1 saved to {local_model_path}")
            local_model_1.to("cpu")
            torch.cuda.empty_cache()
            gc.collect()

            local_model_2.to(device)
            local_model_2 = copy.deepcopy(global_model)
            optimizer_2 = torch.optim.Adam(local_model_2.parameters(), lr=1e-5)

            train(
                model=local_model_2, 
                train_loader=client_2[0], 
                val_loader=client_2[1], 
                optimizer=optimizer_2, 
                num_epochs=num_epochs, 
                patience=patience, 
                device=device
            )

            if round_num in [2, 5, 10, 20]:
                test(local_model_2, client_2[0], device, f"{root}/global_{round_num}_train_local_{pair_num+1}_client2")
                test(local_model_2, client_2[1], device, f"{root}/global_{round_num}_valid_local_{pair_num+1}_client2")
                test(local_model_2, test_loader, device, f"{root}/global_{round_num}_test_local_{pair_num+1}_client2")
                test(local_model_2, ex_loader_1, device, f"{root}/global_{round_num}_ex1_local_{pair_num+1}_client2")
                test(local_model_2, ex_loader_2, device, f"{root}/global_{round_num}_ex2_local_{pair_num+1}_client2")
                test(local_model_2, ex_loader_3, device, f"{root}/global_{round_num}_ex3_local_{pair_num+1}_client2")

                local_model_path = os.path.join(root, f"{pair_num+1}_client2.pth")
                torch.save(local_model_2.state_dict(), local_model_path)
                print(f"Local model for Client 2 saved to {local_model_path}")
            local_model_2.to("cpu")

            client_weights.append(copy.deepcopy(local_model_1.state_dict()))
            client_weights.append(copy.deepcopy(local_model_2.state_dict()))
            ipa_local_avg_weights = average_weights(client_weights)
            global_model.load_state_dict(ipa_local_avg_weights)

            del local_model_1, local_model_2
            torch.cuda.empty_cache()
            gc.collect()

        global_model.to(device)
        _, acc2 = validate(global_model, test_loader, device)
        global_model.to('cpu')

        global_weights = average_weights_with_ratio([ipa_local_avg_weights, local_avg_weights], [acc1, acc2])
        global_model.load_state_dict(global_weights)

        if round_num in [2, 5, 10, 20]:
            global_model.to(device)
            test(global_model, test_loader, device, f"{root}/global_{round_num}_test_Global")
            test(global_model, ex_loader_1, device, f"{root}/global_{round_num}_ex1_Global")
            test(global_model, ex_loader_2, device, f"{root}/global_{round_num}_ex2_Global")
            test(global_model, ex_loader_3, device, f"{root}/global_{round_num}_ex3_Glboal")
            global_model.to("cpu")


        weights_for_feed = global_weights
        acc1 = acc2

        print(f"=== End of Federated Round {round_num} ===\n")
        torch.cuda.empty_cache()
        gc.collect()

    return global_model

def imp_ipa_federated_train_feedback_performance(global_model, clients, num_rounds, num_epochs, patience, device, root, test_loader, ex_loader_1, ex_loader_2, ex_loader_3):
    global_model.to('cpu')
    paired_clients = [(clients[i], clients[i+1]) for i in range(0, len(clients), 2)]
    weights_for_feed = copy.deepcopy(global_model.state_dict())
    global_model.to(device)
    _, acc1 = validate(global_model, test_loader, device)
    global_model.to('cpu')

    for round_num in range(1, num_rounds + 1):
        print(f"\n=== Federated Round {round_num} / {num_rounds} ===")

        for pair_num, (client_1, client_2) in enumerate(paired_clients):
            client_weights = []
            accs = []
            print(f"\n[Pair {pair_num + 1}] 로컬 학습 시작...")

            local_model_1 = copy.deepcopy(global_model)
            local_model_1.to(device)
            optimizer_1 = torch.optim.Adam(local_model_1.parameters(), lr=1e-5)

            train(
                model=local_model_1, 
                train_loader=client_1[0], 
                val_loader=client_1[1], 
                optimizer=optimizer_1, 
                num_epochs=num_epochs, 
                patience=patience, 
                device=device
            )

            _, acc = validate(local_model_1, test_loader, device)
            accs.append(acc)
            if round_num in [2, 5, 10, 20]:
                test(local_model_1, client_1[0], device, f"{root}/global_{round_num}_train_local_{pair_num+1}_client1")
                test(local_model_1, client_1[1], device, f"{root}/global_{round_num}_valid_local_{pair_num+1}_client1")
                test(local_model_1, test_loader, device, f"{root}/global_{round_num}_test_local_{pair_num+1}_client1")
                test(local_model_1, ex_loader_1, device, f"{root}/global_{round_num}_ex1_local_{pair_num+1}_client1")
                test(local_model_1, ex_loader_2, device, f"{root}/global_{round_num}_ex2_local_{pair_num+1}_client1")
                test(local_model_1, ex_loader_3, device, f"{root}/global_{round_num}_ex3_local_{pair_num+1}_client1")

                local_model_path = os.path.join(root, f"{pair_num+1}_client1.pth")
                torch.save(local_model_1.state_dict(), local_model_path)
                print(f"Local model for Client 1 saved to {local_model_path}")
            local_model_1.to("cpu")


            local_model_2 = copy.deepcopy(global_model)
            local_model_2.to(device)
            optimizer_2 = torch.optim.Adam(local_model_2.parameters(), lr=1e-5)

            train(
                model=local_model_2, 
                train_loader=client_2[0], 
                val_loader=client_2[1], 
                optimizer=optimizer_2, 
                num_epochs=num_epochs, 
                patience=patience, 
                device=device
            )

            _, acc = validate(local_model_2, test_loader, device)
            accs.append(acc)
            if round_num in [2, 5, 10, 20]:
                test(local_model_2, client_2[0], device, f"{root}/global_{round_num}_train_local_{pair_num+1}_client2")
                test(local_model_2, client_2[1], device, f"{root}/global_{round_num}_valid_local_{pair_num+1}_client2")
                test(local_model_2, test_loader, device, f"{root}/global_{round_num}_test_local_{pair_num+1}_client2")
                test(local_model_2, ex_loader_1, device, f"{root}/global_{round_num}_ex1_local_{pair_num+1}_client2")
                test(local_model_2, ex_loader_2, device, f"{root}/global_{round_num}_ex2_local_{pair_num+1}_client2")
                test(local_model_2, ex_loader_3, device, f"{root}/global_{round_num}_ex3_local_{pair_num+1}_client2")

                local_model_path = os.path.join(root, f"{pair_num+1}_client2.pth")
                torch.save(local_model_2.state_dict(), local_model_path)
                print(f"Local model for Client 2 saved to {local_model_path}")
            local_model_2.to("cpu")
            client_weights.append(copy.deepcopy(local_model_1.state_dict()))
            client_weights.append(copy.deepcopy(local_model_2.state_dict()))
            ipa_local_avg_weights = average_weights_with_ratio(client_weights, accs)
            global_model.load_state_dict(ipa_local_avg_weights)
            del local_model_1, local_model_2
            torch.cuda.empty_cache()
            gc.collect()

        global_model.to(device)
        _, acc2 = validate(global_model, test_loader, device)
        global_model.to('cpu')
        global_weights = average_weights_with_ratio([ipa_local_avg_weights, local_avg_weights], [acc1, acc2])
        global_model.load_state_dict(global_weights)

        if round_num in [2, 5, 10, 20]:
            global_model.to(device)
            test(global_model, test_loader, device, f"{root}/global_{round_num}_test_Global")
            test(global_model, ex_loader_1, device, f"{root}/global_{round_num}_ex1_Global")
            test(global_model, ex_loader_2, device, f"{root}/global_{round_num}_ex2_Global")
            test(global_model, ex_loader_3, device, f"{root}/global_{round_num}_ex3_Glboal")
            global_model.to("cpu")

        weights_for_feed = global_weights
        acc1 = acc2

        print(f"=== End of Federated Round {round_num} ===\n")
        torch.cuda.empty_cache()
        gc.collect()

    return global_model