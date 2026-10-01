"""Regression tests for pre-ranking relevance and constraint filtering."""

import unittest
import json
from unittest.mock import patch
from types import SimpleNamespace

import httpx
import numpy as np
from scipy.sparse import csr_matrix

import app.model_loader as ml_module
from app.chatbot import build_db_miss_notice, extract_requirements_heuristic
from app.model_loader import RecommenderService
from app.routers import chat
from app.schemas import InteractionCreate, RecommendedProduct
from app import gemini
from app.routers.interactions import record_interaction
from app.user_context import resolve_user_context


class RelevanceFilteringTests(unittest.TestCase):
    def setUp(self):
        self.metadata = {
            "capture": {
                "title": "4K HDMI Capture Card",
                "features": "120 fps",
                "description": "",
                "categories": "Electronics Video Capture Cards",
                "price": 80,
            },
            "monitor": {
                "title": "4K Gaming Monitor",
                "features": "",
                "description": "",
                "categories": "Electronics Monitors",
                "price": 150,
            },
            "dji_generic": {
                "title": "DJI Osmo Action Camera Accessory Kit",
                "features": "Compatible with DJI Osmo Action; includes 5 mounts",
                "description": "",
                "categories": "Electronics Camera Accessories",
                "price": 30,
            },
            "dji_action_5": {
                "title": "DJI Osmo Action 5 Waterproof Case",
                "features": "",
                "description": "",
                "categories": "Electronics Camera Accessories",
                "price": 35,
            },
        }
        self.recommender = object.__new__(RecommenderService)
        self.recommender._loaded = True
        self.recommender._model = object()
        self.recommender._alpha = 0.7
        self.recommender._meta_by_asin = self.metadata
        self.recommender.cbf_scores_for_query = lambda query: {
            "capture": 0.8,
            "monitor": 0.95,
        }
        self.recommender.cf_scores_for_user = lambda user: {
            "capture": 1.0,
            "monitor": 5.0,
        }

    def test_hybrid_only_ranks_constraint_valid_candidates(self):
        eligible, _ = self.recommender.relevant_candidate_asins(
            "4k capture card",
            target_type="capture card",
            max_price=100,
            category="capture",
        )

        ranked = self.recommender.hybrid_scores_for_query(
            user_key="demo",
            query="4k capture card",
            candidate_asins=eligible,
        )

        self.assertEqual(eligible, {"capture"})
        self.assertEqual([item["asin"] for item in ranked], ["capture"])

    def test_specific_dji_generation_rejects_generic_osmo_products(self):
        self.recommender.cbf_scores_for_query = lambda query: {
            "dji_generic": 0.9,
            "dji_action_5": 0.7,
        }
        eligible, unmatched = self.recommender.relevant_candidate_asins(
            "DJI Osmo Action 5 camera accessories",
            target_type="camera accessories",
            requirements=["dji osmo action 5"],
            category="camera",
        )

        self.assertEqual(eligible, {"dji_action_5"})
        self.assertEqual(unmatched, [])

    def test_specific_dji_generation_absence_is_reported(self):
        self.recommender.cbf_scores_for_query = lambda query: {"dji_generic": 0.9}
        eligible, unmatched = self.recommender.relevant_candidate_asins(
            "DJI Osmo Action 5 camera accessories",
            target_type="camera accessories",
            requirements=["dji osmo action 5"],
            category="camera",
        )

        self.assertEqual(eligible, set())
        self.assertEqual(unmatched, ["dji osmo action 5"])

    def test_only_active_database_products_can_become_candidates(self):
        self.recommender.cbf_scores_for_query = lambda query: {
            "capture": 0.8,
            "monitor": 0.95,
        }
        db_metadata = {
            "capture": self.metadata["capture"],
        }

        eligible, _ = self.recommender.relevant_candidate_asins(
            "4k capture card",
            target_type="capture card",
            catalog_metadata=db_metadata,
        )

        self.assertEqual(eligible, {"capture"})

    def test_requirement_coverage_uses_database_product_metadata(self):
        self.recommender.cbf_scores_for_query = lambda query: {"dji_generic": 0.9}
        db_metadata = {
            "dji_generic": {
                "title": "DJI Osmo Action 5 Camera Accessory Kit",
                "features": "",
                "description": "",
                "categories": "Electronics Camera Accessories",
                "price": 30,
            },
        }

        eligible, unmatched = self.recommender.relevant_candidate_asins(
            "DJI Osmo Action 5 camera accessories",
            target_type="camera accessories",
            requirements=["dji osmo action 5"],
            catalog_metadata=db_metadata,
        )

        self.assertEqual(eligible, {"dji_generic"})
        self.assertEqual(unmatched, [])

    def test_no_catalog_answer_uses_constraint_aware_llm_prompt(self):
        captured = {}

        def fake_generate(question, **kwargs):
            captured["question"] = question
            captured.update(kwargs)
            return "Consider a verified-compatible camera accessory and confirm the model before buying."

        with patch.object(gemini, "_generate", side_effect=fake_generate):
            answer = gemini.answer_no_catalog_match(
                "DJI Osmo Action 5 camera accessories",
                requirements=["dji osmo action 5", "underwater"],
                max_price=50,
                suggestions=[{"title": "DJI Osmo Action 5 Waterproof Case"}],
                rating_history=[{"title": "Camera Sling Bag", "rating": 5.0}],
            )

        self.assertIn("dji osmo action 5", captured["question"])
        self.assertIn("underwater", captured["question"])
        self.assertIn("Maximum budget: $50.00", captured["question"])
        self.assertIn("DJI Osmo Action 5 Waterproof Case", captured["question"])
        self.assertIn("Camera Sling Bag: 5.0/5", captured["question"])
        self.assertIn("Do not say you lack access", captured["question"])
        self.assertIn("no sufficiently relevant product", captured["system_prompt"])
        self.assertTrue(answer)

    def test_dynamic_llm_product_terms_drive_unknown_product_filter(self):
        self.metadata["perimeter_camera"] = {
            "title": "Outdoor Perimeter Surveillance Camera",
            "features": "",
            "description": "",
            "categories": "Electronics",
            "price": 80,
        }
        self.metadata["mount"] = {
            "title": "Outdoor Perimeter Camera Wall Mount",
            "features": "",
            "description": "",
            "categories": "Electronics",
            "price": 15,
        }
        valid, _ = self.recommender.filter_and_validate_candidates(
            [{"asin": "perimeter_camera"}, {"asin": "mount"}],
            target_type="perimeter monitoring unit",
            matching_terms=["outdoor perimeter surveillance camera"],
            exclusions=["camera wall mount"],
        )

        self.assertEqual([item["asin"] for item in valid], ["perimeter_camera"])

    def test_type_synonyms_do_not_match_arbitrary_word_prefixes(self):
        self.assertIsNone(ml_module._canonical_type("perimeter monitoring unit"))

    def test_inferred_exclusions_cannot_remove_requested_type_aliases(self):
        exclusions = chat._merge_type_exclusions(
            "camera bag",
            ["camera bag", "camera backpack"],
            ["camera", "camera backpack", "camera strap", "lens"],
            ["wall mount"],
        )

        self.assertEqual(exclusions, ["camera strap", "lens", "wall mount"])

    def test_equivalent_memory_card_exclusion_is_removed(self):
        exclusions = chat._merge_type_exclusions(
            "microsd card",
            ["micro sd card"],
            ["sd card", "card reader"],
            [],
            self.recommender.canonical_product_type,
        )

        self.assertEqual(exclusions, ["card reader"])

    def test_llm_extraction_returns_dynamic_search_terms_and_exclusions(self):
        parsed = {
            "intent": "product_search",
            "search_query": "outdoor perimeter camera",
            "product_type": "perimeter monitoring unit",
            "max_price": 400,
            "matching_terms": ["outdoor perimeter surveillance camera"],
            "requirements": ["2k resolution"],
                "preferences": ["outdoor use"],
            "excluded_types": ["wall mount"],
        }
        with patch.object(gemini, "_generate", return_value=json.dumps(parsed)):
            result = gemini.extract_requirements("find an outdoor perimeter monitoring unit")

        self.assertEqual(result["search_query"], "outdoor perimeter camera")
        self.assertEqual(result["max_price"], 400.0)
        self.assertEqual(result["matching_terms"], ["outdoor perimeter surveillance camera"])
        self.assertEqual(result["preferences"], ["outdoor use"])
        self.assertEqual(result["excluded_types"], ["wall mount"])

    def test_llm_nlu_accepts_greetings_and_questions_without_product_slots(self):
        for intent, message in (
            ("greeting", "hello there"),
            ("electronics_question", "How does image stabilization work?"),
        ):
            parsed = {
                "domain": "electronics",
                "intent": intent,
                "search_query": "",
                "product_type": "",
                "category": None,
                "max_price": None,
                "personalization_requested": False,
                "matching_terms": [],
                "requirements": [],
                "excluded_types": [],
            }
            with patch.object(gemini, "_generate", return_value=json.dumps(parsed)):
                result = gemini.extract_requirements(message)

            self.assertIsNotNone(result)
            self.assertEqual(result["intent"], intent)

    def test_camera_bag_is_a_distinct_accessory_product_type(self):
        self.metadata["camera_bag"] = {
            "title": "Lowepro Adventura Camera Shoulder Bag",
            "features": "",
            "description": "",
            "categories": "Electronics Camera Bags",
            "price": 45,
        }
        self.metadata["camera_mount"] = {
            "title": "Camera Wall Mount Bracket",
            "features": "",
            "description": "",
            "categories": "Electronics Camera Mounts",
            "price": 15,
        }
        valid, _ = self.recommender.filter_and_validate_candidates(
            [{"asin": "camera_bag"}, {"asin": "camera_mount"}],
            target_type="camera bag",
        )
        ids = {item["asin"] for item in valid}
        self.assertIn("camera_bag", ids)
        self.assertNotIn("camera_mount", ids)

    def test_excluded_type_in_included_features_does_not_reject_product(self):
        self.metadata["camera_bag_with_tripod"] = {
            "title": "Camera Backpack Bag with Tripod Holder",
            "features": "",
            "description": "",
            "categories": "Electronics Camera Bags",
            "price": 45,
        }
        valid, _ = self.recommender.filter_and_validate_candidates(
            [{"asin": "camera_bag_with_tripod"}],
            target_type="camera bag",
            exclusions=["tripod"],
        )

        self.assertEqual([item["asin"] for item in valid], ["camera_bag_with_tripod"])

    def test_previous_ratings_phrase_is_removed_from_catalog_query(self):
        from app.chatbot import parse_message

        parsed = parse_message("recommend camera bags based on my previous ratings")
        self.assertEqual(parsed["query"], "camera bags")

    def test_accessory_request_rejects_host_device(self):
        self.metadata["gopro_camera"] = {
            "title": "GoPro HERO12 Black Waterproof Action Camera",
            "features": "",
            "description": "",
            "categories": "Electronics Camera",
            "price": 350,
        }
        self.metadata["gopro_case"] = {
            "title": "GoPro Protective Housing Case for HERO12",
            "features": "",
            "description": "",
            "categories": "Electronics Camera Accessories",
            "price": 50,
        }
        self.recommender.cbf_scores_for_query = lambda query: {
            "gopro_camera": 0.95,
            "gopro_case": 0.70,
        }

        eligible, _ = self.recommender.relevant_candidate_asins(
            "camera accessories",
            target_type="camera accessories",
        )

        self.assertIn("gopro_case", eligible)
        self.assertNotIn("gopro_camera", eligible)

    def test_specific_accessory_does_not_fall_back_to_generic_mounts(self):
        self.metadata["media_mod"] = {
            "title": "GoPro Media Mod for HERO12 Black",
            "features": "built-in mic and HDMI",
            "description": "",
            "categories": "Electronics Camera Accessories",
            "price": 80,
        }
        self.metadata["helmet_mount"] = {
            "title": "GoPro Vented Helmet Strap Mount",
            "features": "",
            "description": "",
            "categories": "Electronics Camera Accessories",
            "price": 12,
        }
        self.recommender.cbf_scores_for_query = lambda query: {
            "media_mod": 0.70,
            "helmet_mount": 0.80,
        }

        eligible, unmatched = self.recommender.relevant_candidate_asins(
            "gopro media mod",
            target_type="gopro media mod",
            requirements=["hero 12"],
        )

        self.assertEqual(eligible, {"media_mod"})
        self.assertEqual(unmatched, [])

    def test_requirement_merge_keeps_models_and_drops_vague_use_case(self):
        merged = chat._merge_requirements(["media mod"], ["hero 12", "recording videos"])
        self.assertIn("hero 12", merged)
        self.assertIn("media mod", merged)
        self.assertNotIn("recording videos", merged)

    def test_heuristic_target_type_drops_model_tokens(self):
        reqs = extract_requirements_heuristic(
            "I need a GoPro Hero 12 media mod for recording videos"
        )
        self.assertEqual(reqs["product_type"], "gopro media mod")
        self.assertIn("hero 12", reqs["requirements"])

    def test_budget_words_never_enter_target_type_or_requirements(self):
        reqs = extract_requirements_heuristic("recommend wireless headphones under $50")
        self.assertEqual(reqs["product_type"], "headphone")

        self.assertEqual(
            chat._sanitize_target("recommend wireless headphones under $50"),
            "wireless headphones",
        )
        self.assertEqual(
            chat._merge_requirements(["under $50", "wireless"], []),
            ["wireless"],
        )

    def test_db_miss_notice_mentions_database_and_llm_fallback(self):
        notice = build_db_miss_notice(["hero 12"], has_suggestions=True).lower()
        self.assertIn("database", notice)
        self.assertIn("llm", notice)
        self.assertIn("hero 12", notice)

    def test_binary_flow_hybrid_only_when_candidates_else_none(self):
        """The flow is binary: enough valid DB candidates -> hybrid ranking;
        otherwise NO catalog products at all (LLM fallback handles the reply)."""
        # 6 valid candidates -> hybrid path runs
        self.recommender.cbf_scores_for_query = lambda query: {
            "capture": 0.8, "monitor": 0.95, "x1": 0.5, "x2": 0.5, "x3": 0.5, "x4": 0.5, "x5": 0.5,
        }
        self.metadata.update({f"x{i}": self.metadata["capture"] | {"title": f"Capture Card {i}"}
                              for i in range(1, 6)})
        eligible, _ = self.recommender.relevant_candidate_asins(
            "capture card", target_type="capture card",
        )
        ranked = self.recommender.hybrid_scores_for_query(
            user_key="demo", query="capture card", candidate_asins=eligible,
        )
        self.assertGreaterEqual(len(ranked), 5)
        self.assertTrue(all(r["hybrid_score"] > 0 for r in ranked))

        # 0 valid candidates -> no catalog products; LLM fallback takes over.
        self.recommender.cbf_scores_for_query = lambda query: {"monitor": 0.95}
        eligible, unmatched = self.recommender.relevant_candidate_asins(
            "capture card", target_type="capture card",
        )
        self.assertEqual(eligible, set())
        self.assertTrue(unmatched or not eligible)


class ProductClassificationTests(unittest.TestCase):
    """Structured classification must be flexible, but still controlled."""

    def setUp(self):
        self.metadata = {
            "card": {
                "title": "SanDisk 128GB MicroSDXC Memory Card",
                "features": "U3, supports 4K UHD",
                "description": "",
                "categories": "Electronics Memory Cards",
                "price": 20,
            },
            "case": {
                "title": "GoPro Protective Housing Case for HERO12",
                "features": "",
                "description": "",
                "categories": "Electronics Camera Accessories",
                "price": 50,
            },
            "kit": {
                "title": "Generic Accessory Kit for GoPro Hero Cameras",
                "features": "",
                "description": "",
                "categories": "Electronics Camera Accessories",
                "price": 25,
            },
            "camera": {
                "title": "GoPro HERO12 Black Waterproof Action Camera",
                "features": "",
                "description": "",
                "categories": "Electronics Camera",
                "price": 350,
            },
            "food": {
                "title": "Maxi Tuna Fish Crackers",
                "features": "",
                "description": "",
                "categories": "Electronics Memory Cards",  # mis-categorised on purpose
                "price": 5,
            },
            "security_camera": {
                "title": "Indoor WiFi Security Camera",
                "features": "1080p video",
                "description": "",
                "categories": "Electronics Security Cameras",
                "price": 40,
            },
            "security_camera_mount": {
                "title": "Adjustable Wall Mount for Security Camera",
                "features": "",
                "description": "",
                "categories": "Electronics Camera Mounts",
                "price": 12,
            },
            "micro_sd_reader": {
                "title": "USB MicroSD Card Reader",
                "features": "Reads MicroSDXC cards",
                "description": "",
                "categories": "Electronics Card Readers",
                "price": 8,
            },
        }
        self.rec = object.__new__(RecommenderService)
        self.rec._loaded = True
        self.rec._model = object()
        self.rec._alpha = 0.7
        self.rec._meta_by_asin = self.metadata

    def _filter(self, target, reqs=None):
        return self.rec.filter_and_validate_candidates(
            [{"asin": a} for a in self.metadata],
            target_type=target,
            requirements=reqs or [],
        )

    def test_compat_and_spec_requirements_are_not_hard_filters(self):
        query = self.rec.classify_query("gopro compatible 4k microsd card", ["gopro compatible", "4k"])
        self.assertEqual(query["desired_type"], "memory card")
        self.assertEqual(query["compat"], {"gopro"})
        self.assertEqual(query["hard_reqs"], [])
        self.assertIn("4k", query["soft_specs"])

    def test_universal_component_passes_without_brand_mention(self):
        valid, unmatched = self._filter("gopro compatible 4k microsd card", ["gopro compatible", "4k"])
        ids = {v["asin"] for v in valid}
        self.assertIn("card", ids)
        self.assertNotIn("food", ids)   # title evidence beats mis-categorised category
        self.assertNotIn("camera", ids)  # a host device is not a component
        self.assertEqual(unmatched, [])

    def test_accessory_request_accepts_accessory_kit_and_rejects_host(self):
        valid, _ = self._filter("gopro accessories")
        ids = {v["asin"] for v in valid}
        self.assertIn("kit", ids)
        self.assertIn("case", ids)
        self.assertNotIn("camera", ids)
        self.assertNotIn("food", ids)

    def test_device_fit_accessory_must_declare_the_brand(self):
        self.metadata["generic_case"] = {
            "title": "Universal Waterproof Camera Case",
            "features": "",
            "description": "",
            "categories": "Electronics Camera Accessories",
            "price": 15,
        }
        valid, _ = self._filter("gopro case")
        ids = {v["asin"] for v in valid}
        self.assertIn("case", ids)
        self.assertNotIn("generic_case", ids)

    def test_model_constraint_still_forces_fallback_when_absent(self):
        valid, unmatched = self._filter("gopro media mod", ["hero 12"])
        self.assertEqual(valid, [])
        self.assertEqual(unmatched, ["hero 12"])

    def test_capacity_requirement_is_hard_and_rejects_card_reader(self):
        self.metadata["card_256"] = {
            "title": "SanDisk 256GB MicroSDXC Memory Card",
            "features": "",
            "description": "",
            "categories": "Electronics Memory Cards",
            "price": 25,
        }
        self.metadata["reader_256"] = {
            "title": "USB MicroSD Card Reader with 256GB Card Support",
            "features": "",
            "description": "",
            "categories": "Electronics Card Readers",
            "price": 10,
        }
        self.metadata["camera_256_recording"] = {
            "title": "Amcrest 256GB MicroSD Recording Security Camera",
            "features": "",
            "description": "",
            "categories": "Electronics Camera & Photo Video Surveillance Surveillance Cameras Bullet Cameras",
            "price": 90,
        }
        self.metadata["sdxc_not_microsd"] = {
            "title": "SanDisk 256GB Ultra SDXC Memory Card",
            "features": "",
            "description": "",
            "categories": "Electronics Memory Cards",
            "price": 25,
        }
        valid, unmatched = self.rec.filter_and_validate_candidates(
            [
                {"asin": "card"},
                {"asin": "card_256"},
                {"asin": "reader_256"},
                {"asin": "camera_256_recording"},
            ],
            target_type="microSD card",
            requirements=["256gb"],
            exclusions=["card reader"],
        )

        self.assertEqual([item["asin"] for item in valid], ["card_256"])
        self.assertEqual(unmatched, [])
        micro_valid, _ = self.rec.filter_and_validate_candidates(
            [{"asin": "card_256"}],
            target_type="microSD card",
            requirements=["256gb"],
            matching_terms=["microsd card", "microsdxc card"],
        )
        self.assertEqual([item["asin"] for item in micro_valid], ["card_256"])

    def test_capacity_mentioned_in_camera_title_is_not_a_memory_card(self):
        self.metadata["camera_256_recording"] = {
            "title": "Amcrest 256GB MicroSD Recording Security Camera",
            "features": "",
            "description": "",
            "categories": "Electronics Security Cameras",
            "price": 90,
        }
        valid, _ = self.rec.filter_and_validate_candidates(
            [{"asin": "camera_256_recording"}],
            target_type="microSD card",
            requirements=["256gb"],
        )

        self.assertEqual(valid, [])

    def test_micro_sd_request_rejects_full_size_sd_card(self):
        self.metadata["sdxc_not_microsd"] = {
            "title": "SanDisk 256GB Ultra SDXC Memory Card",
            "features": "",
            "description": "",
            "categories": "Electronics Memory Cards",
            "price": 25,
        }
        valid, _ = self.rec.filter_and_validate_candidates(
            [{"asin": "sdxc_not_microsd"}],
            target_type="microSD card",
            requirements=["256gb"],
            matching_terms=["microsd card", "micro sd card"],
        )

        self.assertEqual(valid, [])

    def test_capacity_requirement_with_separated_unit_matches_catalog_format(self):
        self.assertTrue(
            self.rec._matches_requirement("SanDisk 256 GB MicroSDXC card", "256gb")
        )
        self.assertTrue(
            self.rec._matches_requirement("SanDisk 256GB MicroSDXC card", "256gb")
        )

    def test_capacity_requirement_with_natural_language_suffix_is_hard(self):
        self.assertTrue(
            self.rec._matches_requirement(
                "SanDisk 256 GB MicroSDXC card", "256gb capacity"
            )
        )
        classification = self.rec.classify_query("microSD card", ["256gb capacity"])
        self.assertEqual(classification["hard_reqs"], ["256gb capacity"])

    def test_security_camera_request_rejects_camera_mount(self):
        valid, _ = self._filter("security camera")
        ids = {item["asin"] for item in valid}
        self.assertIn("security_camera", ids)
        self.assertNotIn("security_camera_mount", ids)

    def test_microsd_card_request_rejects_card_reader(self):
        valid, _ = self._filter("microSD card")
        ids = {item["asin"] for item in valid}
        self.assertIn("card", ids)
        self.assertNotIn("micro_sd_reader", ids)

    def test_explicit_product_exclusion_is_applied(self):
        exclusions = self.rec.extract_explicit_exclusions(
            "security camera, not a mount"
        )
        self.assertIn("mount", exclusions)
        valid, _ = self.rec.filter_and_validate_candidates(
            [{"asin": "security_camera"}, {"asin": "security_camera_mount"}],
            target_type="security camera",
            exclusions=exclusions,
        )
        self.assertEqual([item["asin"] for item in valid], ["security_camera"])


class CatalogFieldContractTests(unittest.TestCase):
    """The catalog keeps its ORIGINAL fields only - no derived per-product
    classification (product_type/accessory_type/model/compatible_products/
    keywords) is computed, stored, or exposed."""

    def test_no_per_product_classifier_exists(self):
        import app.model_loader as ml

        self.assertFalse(hasattr(RecommenderService, "_profile"))
        self.assertFalse(hasattr(RecommenderService, "classify_product"))
        self.assertFalse(hasattr(ml, "_classify_meta"))

    def test_matching_reads_only_original_catalog_fields(self):
        rec = object.__new__(RecommenderService)
        rec._loaded = True
        rec._model = object()
        rec._alpha = 0.7
        rec._meta_by_asin = {
            "A": {"title": "SanDisk 128GB MicroSDXC", "description": "", "categories": "Electronics", "store": "SanDisk", "price": 20},
        }
        self.assertIn("sandisk 128gb microsdxc", ml_module._product_text(rec._meta_by_asin["A"]))

    def test_cf_block_is_absolutely_anchored_not_relative(self):
        rec = object.__new__(RecommenderService)
        rec._alpha = 0.7
        # User predictions: A=5.0 (great), B=2.0 (mediocre)
        rec.cf_scores_for_user = lambda user: {"A": 5.0, "B": 2.0}
        block = rec.cf_block_for("u", ["A", "B", "C"])
        self.assertAlmostEqual(block[0], 1.0)   # 5.0/4.0 clipped to 1.0
        self.assertAlmostEqual(block[1], 0.5)   # 2.0/4.0 - mediocre stays mediocre
        self.assertAlmostEqual(block[2], 0.875)  # missing -> user mean 3.5 / 4.0


class RecommendedProductSourceTests(unittest.TestCase):
    def test_llm_source_is_preserved_and_defaults_to_hybrid(self):
        llm = RecommendedProduct(
            asin="LLM-1", title="X", category=None, price=None, rating=None,
            rating_count=0, cf_score=0.0, cbf_score=0.0, hybrid_score=0.0,
            rank=1, source="llm",
        )
        catalog = RecommendedProduct(
            asin="B1", title="Y", category=None, price=1.0, rating=4.0,
            rating_count=1, cf_score=0.1, cbf_score=0.2, hybrid_score=0.3, rank=1,
        )
        self.assertEqual(llm.source, "llm")
        self.assertEqual(catalog.source, "hybrid")


class NewUserRatingHistoryTests(unittest.TestCase):
    def test_cold_start_scores_use_rating_direction(self):
        recommender = object.__new__(RecommenderService)
        recommender._alpha = 0.5
        recommender._model = SimpleNamespace(
            cf=SimpleNamespace(
                item_to_idx={"liked": 0, "disliked": 1, "other": 2},
                Vt=np.array([[1.0, -1.0, 0.5]]),
            ),
            cbf=SimpleNamespace(
                item_to_idx={"liked": 0, "disliked": 1, "other": 2},
                item_ids=["liked", "disliked", "other"],
                item_tfidf_matrix=csr_matrix(
                    [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
                ),
            ),
        )

        positive = recommender.cold_start_scores(
            ["liked", "disliked"], {"liked": 5.0, "disliked": 1.0}
        )
        reversed_ratings = recommender.cold_start_scores(
            ["liked", "disliked"], {"liked": 1.0, "disliked": 5.0}
        )

        self.assertGreater(positive["liked"], positive["disliked"])
        self.assertGreater(reversed_ratings["disliked"], reversed_ratings["liked"])

    def test_user_context_returns_latest_saved_review_ratings(self):
        rows = [
            ("review", "liked", 5.0),
            ("review", "disliked", 1.0),
            ("review", "liked", 2.0),
        ]

        class FakeQuery:
            def __init__(self, result_rows):
                self.result_rows = result_rows

            def filter(self, *args):
                return self

            def join(self, *args):
                return self

            def order_by(self, *args):
                return self

            def limit(self, *args):
                return self

            def first(self):
                return None

            def all(self):
                return self.result_rows

        class FakeSession:
            def query(self, *args):
                return FakeQuery(rows if len(args) > 1 else [])

        context = resolve_user_context(FakeSession(), user_id=12)

        self.assertEqual(context.cold_start_ratings, {"liked": 5.0, "disliked": 1.0})
        self.assertIn("liked", context.cold_start_asins)
        self.assertIn("disliked", context.cold_start_asins)

    def test_rerating_updates_existing_review(self):
        product = SimpleNamespace(id=23)
        saved_review = SimpleNamespace(rating=1.0)

        class FakeQuery:
            def filter(self, *args):
                return self

            def order_by(self, *args):
                return self

            def first(self):
                return saved_review

        class FakeSession:
            committed = False

            def get(self, model, product_id):
                return product if product_id == product.id else None

            def query(self, *args):
                return FakeQuery()

            def commit(self):
                self.committed = True

            def refresh(self, row):
                pass

        db = FakeSession()
        updated = record_interaction(
            InteractionCreate(product_id=23, interaction_type="review", rating=5),
            SimpleNamespace(id=7),
            db,
        )

        self.assertIs(updated, saved_review)
        self.assertEqual(saved_review.rating, 5)
        self.assertTrue(db.committed)


class GeminiModelFallbackTests(unittest.TestCase):
    def test_generate_falls_through_quota_exceeded_model(self):
        calls: list[str] = []

        class FakeResp:
            def __init__(self, status: int, payload: dict | None = None):
                self.status_code = status
                self._payload = payload or {}
                self.text = "quota exhausted"

            def json(self):
                return self._payload

            def raise_for_status(self):
                if self.status_code >= 400:
                    raise httpx.HTTPStatusError(
                        "err",
                        request=httpx.Request("POST", "https://example.test"),
                        response=self,
                    )

        def fake_post(url, **kwargs):
            calls.append(url)
            if "gemini-primary" in url:
                return FakeResp(429)
            return FakeResp(
                200,
                {"candidates": [{"content": {"parts": [{"text": "hello from fallback"}]}}]},
            )

        with (
            patch.object(gemini.settings, "gemini_model", "gemini-primary"),
            patch.object(gemini.settings, "gemini_fallback_models", "gemini-secondary"),
            patch.object(gemini.httpx, "post", side_effect=fake_post),
        ):
            out = gemini._generate("hi")

        self.assertEqual(out, "hello from fallback")
        self.assertTrue(any("gemini-secondary" in url for url in calls))


if __name__ == "__main__":
    unittest.main()