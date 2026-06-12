(function () {
    const LOCALE = {
        zh: {
            title: "Anima 画师串调度混合",
            language: "界面语言",
            baseTab: "底图",
            hiresTab: "高分修复",
            templateTab: "模板",
            helpTab: "说明",
        },
        en: {
            title: "Anima Artist Scheduled Mixer",
            language: "UI language",
            baseTab: "Base",
            hiresTab: "Hires. fix",
            templateTab: "Template",
            helpTab: "Guide",
        },
    };

    function normalizedText(node) {
        return (node.textContent || "").replace(/\s+/g, " ").trim();
    }

    function textIs(node, values) {
        const value = normalizedText(node);
        return values.some(function (item) {
            return value === item;
        });
    }

    function setKnownText(root, zhText, enText, target) {
        root.querySelectorAll("label, span, button, [role='tab']").forEach(function (node) {
            if (textIs(node, [zhText, enText])) {
                node.textContent = target;
            }
        });
    }

    function refreshPluginLanguage(language) {
        const lang = LOCALE[language] ? language : "zh";
        const t = LOCALE[lang];
        const app = window.gradioApp ? window.gradioApp() : document;

        app.querySelectorAll('[id$="_anima_artist_scheduled_mixer_accordion"], [id*="_anima_artist_scheduled_mixer_accordion"]').forEach(function (panel) {
            const accordionLabel = panel.querySelector(".label-wrap span, .label-wrap");
            if (accordionLabel && /Artist Scheduled Mixer|画师串调度混合/.test(normalizedText(accordionLabel))) {
                accordionLabel.textContent = t.title;
            }

            setKnownText(panel, LOCALE.zh.language, LOCALE.en.language, t.language);
            setKnownText(panel, LOCALE.zh.baseTab, LOCALE.en.baseTab, t.baseTab);
            setKnownText(panel, LOCALE.zh.hiresTab, LOCALE.en.hiresTab, t.hiresTab);
            setKnownText(panel, LOCALE.zh.templateTab, LOCALE.en.templateTab, t.templateTab);
            setKnownText(panel, LOCALE.zh.helpTab, LOCALE.en.helpTab, t.helpTab);
        });
    }

    function scan() {
        document.querySelectorAll("[data-anima-artist-language-refresh]").forEach(function (marker) {
            if (marker.dataset.animaArtistRefreshReady === "1") return;
            marker.dataset.animaArtistRefreshReady = "1";
            refreshPluginLanguage(marker.dataset.animaArtistLanguageRefresh);
        });
    }

    function start() {
        scan();
        const observer = new MutationObserver(scan);
        observer.observe(document.body, { childList: true, subtree: true });
    }

    if (window.onUiLoaded) {
        window.onUiLoaded(start);
    } else if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", start);
    } else {
        start();
    }
})();
