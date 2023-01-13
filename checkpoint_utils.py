import torch
import os

def save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer, decoder_optimizer,
                    top5, history, is_best):

    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'top5': top5,
             'history': history,
             'encoder': encoder.state_dict(),
             'decoder': decoder.state_dict(),
             'encoder_optimizer': encoder_optimizer.state_dict() if not encoder_optimizer is None else None,
             'decoder_optimizer': decoder_optimizer.state_dict()}
    filename = 'checkpoint_' + data_name + '.pth.tar'
    torch.save(state, filename)

    if is_best:
        torch.save(state, 'BEST_' + filename)
        
def load_checkpoint(data_name, encoder, decoder, encoder_optimizer, decoder_optimizer, best=False):
    isExist = os.path.exists('BEST_checkpoint_' + data_name + '.pth.tar')
    
    if best and isExist:
        checkpoint = torch.load('BEST_checkpoint_' + data_name + '.pth.tar')
    else:
        checkpoint = torch.load('checkpoint_' + data_name + '.pth.tar')
    
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
    if encoder_optimizer is not None:
        encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])
    decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer'])
    
    return encoder, decoder, encoder_optimizer, decoder_optimizer, checkpoint['epoch'], checkpoint['epochs_since_improvement'], checkpoint['history']