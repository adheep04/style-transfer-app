def get_config():
    return {
        'max_steps' : 18000,
        'lr' : 0.2,
        'factor' : 0.75,
        'freq' : 10,
        'alpha' : 1,
        'beta' : 1e-6,
        'trial' : 'style_trial_x',
        'patience' : 6,
        'content_layer' : 2,
        'style_layer' : 5,
        'content_path' : './temp/data/content.jpg',
        'style_path' : './temp/data/style.jpg',
        'content_dir' : './temp/data',
        'style_dir' : './temp/data',
        'wl' : [0.2, 0.2, 0.2, 0.2, 0.2]
    }