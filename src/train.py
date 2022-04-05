import logging

import torch

from src.metrics import MeanClassAccuracy
from src.utils import check_best_loss, check_best_score, evaluate_dataset, save_best_model


def train(train_loader, val_loader, model, criterion, optimizer, lr_scheduler, epochs, device, writer, metrics,
          train_stats, val_stats, log_dir, new_model_attention=False, model_devise=False, apn=False, cjme=False, args=None):
    best_loss = None
    best_score = None

    for epoch in range(epochs):
        train_loss = train_step(train_loader, model, criterion, optimizer, epoch, epochs, writer, device, metrics,
                                train_stats, new_model_attention, model_devise, apn, cjme, args)
        val_loss, val_hm = val_step(val_loader, model, criterion, epoch, epochs, writer, device, metrics, val_stats,
                                     new_model_attention, model_devise,apn,cjme, args)

        best_loss = check_best_loss(epoch, best_loss, val_loss, model, optimizer, log_dir)
        best_score = check_best_score(epoch, best_score, val_hm, model, optimizer, log_dir)

        if args.save_checkpoints:
            # save_best_model(epoch, val_loss, model, optimizer, log_dir / "checkpoints", metric="loss", checkpoint=True)
            save_best_model(epoch, val_hm, model, optimizer, log_dir / "checkpoints", metric="score", checkpoint=True)

        if lr_scheduler:
            lr_scheduler.step(val_hm)
        if  new_model_attention==True:
            model.optimize_scheduler(val_hm)
    return best_loss, best_score

def train_step(data_loader, model, criterion, optimizer, epoch, epochs, writer, device, metrics, stats,
               new_model_attention,model_devise, apn, cjme, args):
    logger = logging.getLogger()
    model.train()

    for metric in metrics:
        metric.reset()

    batch_loss = 0

    for batch_idx, (data, target) in enumerate(data_loader):
        model.train()
        p = data["positive"]
        q = data["negative"]

        x_p_a = p["audio"].to(device)
        x_p_v = p["video"].to(device)
        x_p_t = p["text"].to(device)
        x_p_num = target["positive"].to(device)

        x_q_a = q["audio"].to(device)
        x_q_v = q["video"].to(device)
        x_q_t = q["text"].to(device)

        if new_model_attention==False and model_devise==False and apn==False:
            inputs = (
                x_p_a, x_p_v, x_p_t,
                x_q_a, x_q_v, x_q_t
            )
        elif new_model_attention==True:
            inputs = (
                x_p_a, x_p_v, x_p_num, x_p_t, x_q_a, x_q_v, x_q_t
            )
        else:
            inputs=(
                x_p_a, x_p_v, x_p_num, x_p_t
            )

        if args.z_score_inputs:
            inputs = tuple([(x - torch.mean(x)) / torch.sqrt(torch.var(x)) for x in inputs])

        if new_model_attention==False and model_devise==False and apn==False:
            if cjme==True:
                outputs=model(*inputs)
                embeddings, mapping_dict = data_loader.dataset.zsl_dataset.map_embeddings_target
                embeddings_projected=model.get_classes_embedding(embeddings)
                loss, loss_details=criterion(*outputs, embeddings_projected.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                outputs = model(*inputs)
                loss, loss_details = criterion(*outputs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        elif apn==True:
            embeddings, mapping_dict = data_loader.dataset.zsl_dataset.map_embeddings_target
            for i in range(inputs[2].shape[0]):
                inputs[2][i] = mapping_dict[(inputs[2][[i]]).item()]
            input_features = torch.cat((inputs[1], inputs[0]), 1)
            output_final, pre_attri, attention, pre_class, attribute = model(input_features, embeddings)
            loss, loss_details=criterion(model, output_final, pre_attri, pre_class, inputs[3] , inputs[2])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        elif model_devise==True:
            embeddings, mapping_dict =data_loader.dataset.zsl_dataset.map_embeddings_target
            for i in range(inputs[2].shape[0]):
                inputs[2][i] = mapping_dict[(inputs[2][[i]]).item()]
            input_features=torch.cat((inputs[1], inputs[0]), 1)
            outputs, _, _=model(input_features,embeddings)
            loss, loss_details=criterion(outputs, inputs[2])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        elif new_model_attention==True:
            loss, loss_details = model.optimize_params(*inputs, optimize=True)
            audio_emb, video_emb, emb_cls=model.get_embeddings(inputs[0], inputs[1], inputs[3])
            outputs=torch.stack([video_emb, emb_cls], dim=0)

        batch_loss += loss.item()

        p_target = target["positive"].to(device)
        q_target = target["negative"].to(device)

        # stats
        iteration = len(data_loader) * epoch + batch_idx


    batch_loss /= (batch_idx + 1)
    stats.update((epoch, batch_loss, None))

    logger.info(
        f"TRAIN\t"
        f"Epoch: {epoch}/{epochs}\t"
        f"Iteration: {iteration}\t"
        f"Loss: {batch_loss:.4f}\t"
    )
    return batch_loss



def val_step(data_loader, model, criterion, epoch, epochs, writer, device, metrics, stats,
             new_model_attention,model_devise,apn,cjme, args=None):

    logger = logging.getLogger()
    model.eval()

    for metric in metrics:
        metric.reset()

    with torch.no_grad():
        batch_loss = 0
        hm_score = 0
        for batch_idx, (data, target) in enumerate(data_loader):
            p = data["positive"]
            q = data["negative"]

            x_p_a = p["audio"].to(device)
            x_p_v = p["video"].to(device)
            x_p_t = p["text"].to(device)
            x_p_num = target["positive"].to(device)

            x_q_a = q["audio"].to(device)
            x_q_v = q["video"].to(device)
            x_q_t = q["text"].to(device)

            if new_model_attention==False and model_devise==False and apn==False:
                inputs = (
                    x_p_a, x_p_v, x_p_t,
                    x_q_a, x_q_v, x_q_t
                )
            elif new_model_attention==True:
                inputs = (
                    x_p_a, x_p_v, x_p_num, x_p_t, x_q_a, x_q_v,x_q_t
                )
            else:
                inputs = (
                    x_p_a, x_p_v, x_p_num, x_p_t
                )

            if args.z_score_inputs:
                inputs = tuple([(x - torch.mean(x)) / torch.sqrt(torch.var(x)) for x in inputs])

            if new_model_attention==False and model_devise==False and apn==False:
                if cjme==True:
                    outputs = model(*inputs)
                    embeddings, mapping_dict = data_loader.dataset.zsl_dataset.map_embeddings_target
                    embeddings_projected = model.get_classes_embedding(embeddings)
                    loss, loss_details = criterion(*outputs, embeddings_projected)
                else:
                    outputs = model(*inputs)
                    loss, loss_details = criterion(*outputs)
            elif model_devise==True:
                embeddings, mapping_dict = data_loader.dataset.zsl_dataset.map_embeddings_target
                for i in range(inputs[2].shape[0]):
                    inputs[2][i] = mapping_dict[(inputs[2][[i]]).item()]
                input_features=torch.cat((inputs[1], inputs[0]), 1)
                outputs, _, _=model(input_features, embeddings)
                loss, loss_details=criterion(outputs, inputs[2])
            elif apn == True:
                embeddings, mapping_dict = data_loader.dataset.zsl_dataset.map_embeddings_target
                for i in range(inputs[2].shape[0]):
                    inputs[2][i] = mapping_dict[(inputs[2][[i]]).item()]
                input_features = torch.cat((inputs[1], inputs[0]), 1)
                output_final, pre_attri, attention, pre_class, attribute = model(input_features, embeddings)
                loss, loss_details = criterion(model, output_final, pre_attri, pre_class, inputs[3], inputs[2])
                outputs=output_final
            elif new_model_attention==True:
                loss, loss_details = model.optimize_params(*inputs)
                audio_emb, video_emb, emb_cls = model.get_embeddings(inputs[0], inputs[1], inputs[3])
                outputs = (video_emb, emb_cls)

            batch_loss += loss.item()

            p_target = target["positive"].to(device)
            q_target = target["negative"].to(device)

            # stats
            iteration = len(data_loader) * epoch + batch_idx
            if iteration % len(data_loader) == 0:
                for metric in metrics:
                    metric(outputs, (p_target, q_target), (loss, loss_details))
                    for key, value in metric.value().items():
                        if "recall" in key:
                            continue
                        if "both_hm" in key:
                            hm_score = value
                        if "both_zsl" in key:
                            zsl_score=value
                        writer.add_scalar(
                            f"val_{key}", value, iteration
                        )

        batch_loss /= (batch_idx + 1)
        stats.update((epoch, batch_loss, hm_score))

        logger.info(
            f"VALID\t"
            f"Epoch: {epoch}/{epochs}\t"
            f"Iteration: {iteration}\t"
            f"Loss: {batch_loss:.4f}\t"
            f"ZSL score: {zsl_score:.4f}\t"
            f"HM: {hm_score:.4f}"
        )
    return batch_loss, hm_score
