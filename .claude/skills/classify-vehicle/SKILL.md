---
name: classify-vehicle
description: >
  Agent-in-the-loop vehicle-type curation for auto_sniper_ml, done directly with
  no external API. Invoke after `python scripts/vehicle_type_requests.py` has
  written data/clusters/vehicle_type_requests.json. Reads ambiguous listing
  titles and tags each with a vehicle type so non-cars (motorcycles, boats,
  trailers, equipment) are kept out of the car pricing model. Use when the user
  asks to classify listings, filter non-cars, or "do the vehicle-type step".
---

# Classify vehicle type

The Facebook vehicles feed is not just cars — motorcycles, ATVs, snowmobiles,
jet skis, boats, trailers, campers, RVs and heavy/farm equipment all appear.
Pricing a $7,500 motorcycle against $16k sedans invents a fake "53% under
market" deal. `src/ml/vehicle_type.py` has a keyword rule for the obvious cases;
your job is the ambiguous residue.

## Inputs

`data/clusters/vehicle_type_requests.json`:

```json
{
  "instructions": "...",
  "listings": [
    {"item_id": "1079231111321315", "title": "2018 honda cbr", "price": 7500.0,
     "assigned_label": "Honda Accord", "rule_says_non_car": true}
  ]
}
```

These are listings whose title doesn't clearly match the car model they were
assigned to, or is too terse to tell. `rule_says_non_car` is the keyword rule's
current verdict — confirm or correct it.

## Procedure

1. Read `data/clusters/vehicle_type_requests.json`.
2. For **every** listing, decide the type from the title (and price as a sanity
   check — a "2018 Honda" at $7,500 that says "cbr" is a motorcycle, not an
   Accord):

   | type | keep? | examples |
   |---|---|---|
   | `car` | ✓ | sedan, hatchback, coupe, wagon |
   | `truck` | ✓ | F-150, Silverado, Tacoma, Ram 1500 |
   | `van` | ✓ | Grand Caravan, Sienna, Transit, Odyssey |
   | `suv` | ✓ | RAV4, CR-V, Explorer, 4Runner, Rogue |
   | `motorcycle` | ✗ | CBR, Ninja, Harley, Grom, any "cc" bike, scooter, moped |
   | `atv` | ✗ | quad, side-by-side, UTV, dirt bike |
   | `boat` | ✗ | pontoon, fishing boat, jet ski, Sea-Doo |
   | `trailer` | ✗ | utility / cargo / dump / travel trailer |
   | `equipment` | ✗ | tractor, skid steer, excavator, mower, forklift |
   | `rv` | ✗ | motorhome, fifth wheel, camper |
   | `other` | ✗ | golf cart, snowmobile, parts, anything else |

   - Title with no model and no other signal ("2015 Yamaha", "running $3000") →
     `other` (we can't price it safely).
   - A real car whose title just doesn't match its cluster (e.g. "2013 Honda"
     assigned to "Honda Odyssey" but it's clearly a sedan) → still tag the
     correct type (`car`), the label-mismatch guard in valuation handles the
     pricing exclusion separately.
3. Write / merge `data/clusters/vehicle_type.json`:

   ```json
   { "1079231111321315": "motorcycle", "1348155627072412": "car", ... }
   ```

   One entry per listing in the request (also acceptable:
   `{"types": { ... }}`). If the file exists, merge — don't drop prior entries.
4. Re-run the pipeline (`bash scripts/run_once.sh` or `python -m
   src.ml.run_pipeline --incremental`) so `load_data` drops the non-cars.

## Notes

- This is a curation pass, not a production classifier. Re-run
  `scripts/vehicle_type_requests.py` and this skill whenever a batch of new
  ambiguous listings accumulates.
- `item_id`s are stable — a tag stays correct forever, so the file only grows.
- If the request file is missing, run `python scripts/vehicle_type_requests.py`.
