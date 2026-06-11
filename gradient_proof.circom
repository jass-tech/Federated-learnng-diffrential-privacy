pragma circom 2.0.0;

template GradientVerifier() {
    signal input hash_input;
    signal input update_value;
    
    // Proof: update_value^2 == hash_input^2 + 1
    var temp = hash_input * hash_input;
    update_value * update_value === temp + 1;
}

component main = GradientVerifier();