mutable struct CandidateLetterInfo
    name_index::Int
    len::Int
    mask::UInt16
end

mutable struct CandidateScore
    matches::UInt8
    used::UInt16
    used_exact::UInt16
    len_partial::UInt16
    last_match_letter_index::UInt16
    transposition_count::UInt8
end

function CandidateScore(len::Number)::CandidateScore
    len_partial = UInt16(1024.0 ÷ len)
    return CandidateScore(0, 0, 0, len_partial, 0, 0)
end

function calculate_jaro_winkler(score::CandidateScore, query_partial::UInt16)::Float64
    transpositions = score.transposition_count > score.matches / 2 ? score.transposition_count - 1 : score.transposition_count
    partial = ((score.matches * score.len_partial) + (score.matches * query_partial)) / 1024.0
    jaro = (partial + 1.0 - (transpositions / score.matches)) / 3.0
    l = trailing_ones(UInt16(score.used_exact & 0x000f))
    return jaro + 0.1 * l * (1.0 - jaro)
end

function get_query_mask(cl::Int,query_mask::UInt16,min_match_dist::Int)::UInt16
    match_distance=cl <= 3 || cl ÷ 2 - 1 <= min_match_dist ? min_match_dist : cl ÷ 2 - 1
    for _ in 0:match_distance
        query_mask = query_mask << 1 | query_mask;
        query_mask = query_mask >> 1 | query_mask;
    end
    return query_mask;
end

"""Converts a string to a list of bitmasks"""
function maskify(query::String;
                 space_char::UInt8=0x60)::Vector{Tuple{UInt8, Vector{UInt16}}}
    min_match_dist = let len=ncodeunits(query); len > 3 ? len ÷ 2 - 1 : 0 end

    [(c-space_char,
      [get_query_mask(cl,0x0001 << i,min_match_dist) for cl in 1:16])
    for (i, c) in enumerate(replace(codeunits(query), 0x20 => space_char))]
end

function build_candidate_lookup(names::Vector{String};
                                allcaps::Bool = false)::Vector{Vector{CandidateLetterInfo}}
    # defining the range of all the letters
    firstletter::UInt8 = allcaps ? 0x41 : 0x61
    letters::UnitRange{UInt8} = UnitRange(firstletter,firstletter+0x19)
    
    letter_lookup = Vector{Vector{CandidateLetterInfo}}(undef, 26)
    for (index, letter) in enumerate(letters)
        candidate_infos = Vector{CandidateLetterInfo}()
        for (name_index, name) in enumerate(names)
            mask = UInt16(0)
            for (matching_index_in_name, c) in enumerate(codeunits(name))
                if c == letter
                    mask += UInt16(2)^(matching_index_in_name - 1)
                end
            end
            if mask != 0
                push!(candidate_infos, CandidateLetterInfo(name_index, ncodeunits(name), mask))
            end
        end
        letter_lookup[letter - firstletter + 1] = candidate_infos
    end
    return letter_lookup
end



function score_letter!(candidate_score::CandidateScore, query_mask::UInt16, candidate_mask::UInt16, query_index::Int)
    whole_mask_result = query_mask & candidate_mask
    check_used_result = xor(whole_mask_result | candidate_score.used, candidate_score.used)
    last_match_letter_index = 0x0001 << trailing_zeros(check_used_result)
    mask_result = check_used_result & last_match_letter_index
    is_match_mask = UInt16(trailing_ones(mask_result >> trailing_zeros(mask_result)))
    candidate_score.used |= mask_result
    candidate_score.used_exact |= mask_result & (0x0001 << query_index-1)
    candidate_score.matches += is_match_mask
    candidate_score.transposition_count += UInt8(mask_result - 1 < candidate_score.last_match_letter_index)
    candidate_score.last_match_letter_index |= mask_result
end


function pseudo_jaro_winkler(names_a::Vector{T}, names_b::Vector{T}, min_jaro_winkler::Float32) where T <: AbstractString
    lookup_a_by_name = Dict(name => findall(x -> x == name, names_a) for name in names_a)
    copy!(names_a,sort!(unique(names_a)))
    lookup_a_by_new_id = Dict(i => lookup_a_by_name[name] for (i, name) in enumerate(names_a))

    lookup_b_by_name = Dict(name => findall(x -> x == name, names_b) for name in names_b)
    copy!(names_b,sort!(unique(names_b)))
    lookup_b_by_new_id = Dict(i => lookup_b_by_name[name] for (i, name) in enumerate(names_b))
    
    base_candidate_lookup = build_candidate_lookup(names_b)
    base_candidate_scores = [CandidateScore(ncodeunits(name)) for name in names_b]
    
    Threads.@threads for (new_a_id, query_name) in collect(enumerate(names_a))
        query_masks_lookup = maskify(query_name)
        query_partial = UInt16(1024 ÷ ncodeunits(query_name))
        candidate_scores = deepcopy(base_candidate_scores)

        for (query_index, (letter_index, query_mask_by_candidate_len)) in enumerate(query_masks_lookup)
            for c_info in @inbounds base_candidate_lookup[letter_index]
                candidate_score = candidate_scores[c_info.name_index]
                query_mask = query_mask_by_candidate_len[c_info.len]
                score_letter!(candidate_score, query_mask, c_info.mask, query_index)
            end
        end
        a_ids = lookup_a_by_new_id[new_a_id]
        for (score_i, score) in enumerate(candidate_scores)
            jw = calculate_jaro_winkler(score, query_partial)
            if jw >= min_jaro_winkler
                b_ids = lookup_b_by_new_id[score_i]
                println("$a_ids,$(names_a[new_a_id]),b_ids: $b_ids, $(names_b[score_i]); $jw,$query_name")
            end
        end
    end
end


pseudo_jaro_winkler(repeat(["apples","pear","bananas","apples","pear","bananas"], 10),
                    repeat(["beep", "pearls", "perl", "banans", "pear"], 1),Float32(0.7))

