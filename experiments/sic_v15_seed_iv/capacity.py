"""Derive and audit approximately 2x trainable parameters per hidden layer."""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path


def sequence(value):
    if isinstance(value, dict):
        value = value['fixed']
    return [int(item) for item in value]


def closest_units(target, count):
    """Nearest achievable integer width, with smaller widths winning ties."""
    low, high = 1, 1
    while count(high) < target:
        high *= 2
    while low < high:
        middle = (low + high) // 2
        if count(middle) < target:
            low = middle + 1
        else:
            high = middle
    return min({max(1, low - 1), low}, key=lambda units: (abs(count(units) - target), units))


def doubled_hidden_config(baseline):
    """Preserve layer count and learning settings; size each hidden layer."""
    result = deepcopy(baseline)
    if baseline.get('n_bilstm_layers', 1) != 1 or baseline.get('classifier_rnn_type', 'bigru') != 'bigru':
        raise ValueError('This capacity comparison targets the current one-BiLSTM/BiGRU SICv15 baseline')
    channels, features = 62, 310
    old_gcn = sequence(baseline['gcn_units'])
    new_gcn, old_input, new_input = [], 1, 1
    for units in old_gcn:
        count = (old_input + 1) * units
        scaled = closest_units(2 * count, lambda width: (new_input + 1) * width)
        new_gcn.append(scaled)
        old_input, new_input = units, scaled
    result['gcn_units'] = {'fixed': new_gcn}

    old_gru = int(baseline['spectral_gru_units'])
    old_input, new_input = channels * old_gcn[-1], channels * new_gcn[-1]
    count = 3 * old_gru * (old_input + old_gru + 2)
    new_gru = closest_units(2 * count, lambda width: 3 * width * (new_input + width + 2))
    result['spectral_gru_units'] = new_gru

    old_lstm = int(baseline['bilstm_units'])
    count = 4 * old_lstm * (features + old_lstm + 1)
    new_lstm = closest_units(2 * count, lambda width: 4 * width * (features + width + 1))
    result['bilstm_units'] = new_lstm

    old_classifier = sequence(baseline['classifier_rnn_units'])
    new_classifier = []
    old_input, new_input = old_gru + 2 * old_lstm, new_gru + 2 * new_lstm
    for units in old_classifier:
        count = 3 * units * (old_input + units + 2)
        scaled = closest_units(2 * count, lambda width: 3 * width * (new_input + width + 2))
        new_classifier.append(scaled)
        old_input, new_input = 2 * units, 2 * scaled
    result['classifier_rnn_units'] = {'fixed': new_classifier}
    subject_units = int(baseline.get('subject_hidden_units', 64))
    result['subject_hidden_units'] = closest_units(2 * (old_input + 1) * subject_units,
                                                  lambda width: (new_input + 1) * width)

    # Decoder node seeds, projection widths, and hidden GCN widths must be
    # independent of the encoder to meet each hidden layer's parameter target.
    for branch, old_embedding, new_embedding in (
            ('gcn_gru', old_gru, new_gru), ('bilstm', 2 * old_lstm, 2 * new_lstm)):
        old_projection = old_seed = old_gcn[-1]
        projection = closest_units(2 * (old_embedding + 1) * old_projection,
                                   lambda width: (new_embedding + 1) * width)
        seed = closest_units(2 * (old_projection + 1) * channels * old_seed,
                             lambda width: (projection + 1) * channels * width)
        old_input, new_input = old_seed, seed
        hidden = []
        for units in reversed(old_gcn[:-1]):
            scaled = closest_units(2 * (old_input + 1) * units,
                                   lambda width: (new_input + 1) * width)
            hidden.append(scaled)
            old_input, new_input = units, scaled
        result[f'{branch}_decoder_gcn_units'] = {'fixed': list(reversed(hidden)) + [seed]}
        result[f'{branch}_decoder_projection_units'] = projection
    return result


def load_model_config(path):
    from eegproc.deep_learning.joint_architectures.sic_common.config import expand_cartesian_grid
    config = json.loads(Path(path).read_text())
    candidates, dimensions = expand_cartesian_grid(config)
    if len(candidates) != 1 or dimensions:
        raise ValueError('Each capacity run requires one fixed model configuration')
    return config


def resolved_model_config(config):
    from eegproc.deep_learning.joint_architectures.sic_common.config import expand_cartesian_grid, split_data_hyperparameters
    candidates, dimensions = expand_cartesian_grid(config)
    if len(candidates) != 1 or dimensions:
        raise ValueError('Expected one fixed configuration')
    resolved, _ = split_data_hyperparameters(candidates[0])
    resolved.update(n_channels=62, n_bands=5, n_classes=4, classification_level='window')
    return resolved


def trainable_count(layer):
    return sum(math.prod(int(size) for size in weight.shape) for weight in layer.trainable_weights)


def inventory(model):
    import tensorflow as tf
    hidden, structural = {}, {}
    for index, layer in enumerate(model.graph_encoder.shared_gcn_layers):
        hidden[f'encoder/gcn_{index}'] = trainable_count(layer)
    hidden['encoder/spectral_gru'] = trainable_count(model.graph_encoder.spectral_gru)
    for index, layer in enumerate(model.temporal_bilstms):
        hidden[f'temporal/lstm_{index}/forward'] = trainable_count(layer.forward_layer)
        hidden[f'temporal/lstm_{index}/backward'] = trainable_count(layer.backward_layer)
    for index, layer in enumerate(model.temporal_norms):
        structural[f'temporal/normalization_{index}'] = trainable_count(layer)
    recurrent_index = normalization_index = 0
    for layer in model.trial_recurrent_classifier.layers:
        if isinstance(layer, tf.keras.layers.Bidirectional):
            hidden[f'classifier/gru_{recurrent_index}/forward'] = trainable_count(layer.forward_layer)
            hidden[f'classifier/gru_{recurrent_index}/backward'] = trainable_count(layer.backward_layer)
            recurrent_index += 1
        elif isinstance(layer, tf.keras.layers.LayerNormalization):
            structural[f'classifier/normalization_{normalization_index}'] = trainable_count(layer)
            normalization_index += 1
    hidden['adversary/hidden'] = trainable_count(model.subject_hidden)
    structural['adversary/subject_outputs'] = trainable_count(model.subject_logits_layer)
    structural['classifier/variational_class_head'] = trainable_count(model.vc_target)
    structural['reconstruction/fusion_scalar'] = trainable_count(model.joint_reconstruction_fusion)
    for branch, decoder in (('graph', model.gcn_gru_decoder), ('temporal', model.bilstm_decoder)):
        hidden[f'decoder_{branch}/input_projection'] = trainable_count(decoder.input_projection)
        hidden[f'decoder_{branch}/node_seed_projection'] = trainable_count(decoder.node_seed_projection)
        for index, layer in enumerate(decoder.graph_layers):
            hidden[f'decoder_{branch}/gcn_{index}'] = trainable_count(layer)
        structural[f'decoder_{branch}/graph_outputs'] = trainable_count(decoder.output_graph)
        structural[f'decoder_{branch}/local_outputs'] = trainable_count(decoder.output_local)
    total = trainable_count(model)
    if sum(hidden.values()) + sum(structural.values()) != total:
        raise ValueError('Parameter inventory missed a trainable layer')
    return dict(hidden=hidden, structural=structural, total_trainable=total,
                total_parameters=model.count_params())


def audit_configs(baseline, scaled, *, timesteps=4):
    import numpy as np
    import tensorflow as tf
    from eegproc.deep_learning.joint_architectures.SICModelv15.sic_model import build_sic_model
    counts = []
    for config in (baseline, scaled):
        model = build_sic_model((timesteps, 310), adjacency=np.eye(62, dtype=np.float32),
                                n_subject_classes=12, **resolved_model_config(config))
        counts.append(inventory(model))
        del model
        tf.keras.backend.clear_session()
    old, new = counts
    if old['hidden'].keys() != new['hidden'].keys():
        raise ValueError('Comparison changed the hidden layer set')
    rows = [dict(layer=name, baseline_parameters=count,
                 comparison_parameters=new['hidden'][name], ratio=new['hidden'][name] / count)
            for name, count in old['hidden'].items()]
    if any(abs(row['ratio'] / 2 - 1) > .01 for row in rows):
        raise ValueError('A hidden layer missed the 2x target by more than 1%')
    structural = [dict(layer=name, baseline_parameters=count,
                      comparison_parameters=new['structural'][name], ratio=new['structural'][name] / count)
                  for name, count in old['structural'].items()]
    return dict(target='2x trainable weights and biases per hidden GCN/recurrent/Dense/Conv layer',
                integer_rounding_tolerance_relative_to_target=.01,
                hidden_layers=rows, structural_layers=structural,
                baseline_total_trainable=old['total_trainable'], comparison_total_trainable=new['total_trainable'],
                baseline_total_parameters=old['total_parameters'], comparison_total_parameters=new['total_parameters'],
                inspected_timesteps=timesteps,
                structural_note='Normalization, fixed-shape output heads, Gaussian class parameters, fusion scalar, and fixed graph are not independent hidden widths.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=Path(__file__).with_name('model_config.json'))
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('model_config_hidden_params_x2.json'))
    parser.add_argument('--report', type=Path, default=Path(__file__).with_name('capacity_report.json'))
    args = parser.parse_args()
    baseline = load_model_config(args.baseline)
    scaled = doubled_hidden_config(baseline)
    report = audit_configs(baseline, scaled)
    args.output.write_text(json.dumps(scaled, indent=2) + '\n')
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
