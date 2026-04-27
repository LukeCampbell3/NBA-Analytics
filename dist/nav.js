(() => {
    const body = document.body;
    const root = document.documentElement;
    const menuBtn = document.getElementById('siteMenuButton');
    const drawer = document.getElementById('siteNavDrawer');
    const overlay = document.getElementById('siteNavOverlay');
    const closeBtn = document.getElementById('siteNavClose');
    const themeToggleBtn = document.getElementById('themeToggleButton');
    const THEME_KEY = 'sports_site_theme_preference';

    const getPreferredTheme = () => {
        const saved = localStorage.getItem(THEME_KEY);
        if (saved === 'dark' || saved === 'light') return saved;
        return (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) ? 'dark' : 'light';
    };

    const applyTheme = (theme) => {
        const isDark = theme === 'dark';
        body.classList.toggle('theme-dark', isDark);
        root.classList.toggle('theme-dark', isDark);
        window.dispatchEvent(new CustomEvent('themechange', { detail: { theme } }));
        if (themeToggleBtn) {
            themeToggleBtn.textContent = isDark ? 'L' : 'D';
            themeToggleBtn.title = isDark ? 'Switch to light mode' : 'Switch to dark mode';
            themeToggleBtn.setAttribute('aria-label', isDark ? 'Switch to light mode' : 'Switch to dark mode');
        }
    };

    applyTheme(getPreferredTheme());

    if (themeToggleBtn) {
        themeToggleBtn.addEventListener('click', () => {
            const next = body.classList.contains('theme-dark') ? 'light' : 'dark';
            localStorage.setItem(THEME_KEY, next);
            applyTheme(next);
        });
    }

    // Optional drawer wiring (some pages do not include drawer markup).
    if (!menuBtn || !drawer || !overlay) return;

    const openNav = () => {
        drawer.classList.add('open');
        overlay.classList.add('open');
        body.classList.add('nav-open');
        menuBtn.setAttribute('aria-expanded', 'true');
    };

    const closeNav = () => {
        drawer.classList.remove('open');
        overlay.classList.remove('open');
        body.classList.remove('nav-open');
        menuBtn.setAttribute('aria-expanded', 'false');
    };

    const isOpen = () => drawer.classList.contains('open');

    menuBtn.addEventListener('click', () => {
        if (isOpen()) closeNav();
        else openNav();
    });

    if (closeBtn) closeBtn.addEventListener('click', closeNav);
    overlay.addEventListener('click', closeNav);

    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && isOpen()) closeNav();
    });

    // Highlight active page by filename. Supports:
    // /index.html, /about.html, /about/, /about/index.html
    const getCurrentPage = () => {
        const segments = window.location.pathname
            .split('/')
            .filter(Boolean)
            .map((part) => part.toLowerCase());

        if (!segments.length) return { page: 'index.html', atSiteRoot: true };

        const last = segments[segments.length - 1];
        if (last.endsWith('.html')) {
            if (last === 'index.html' && segments.length > 1) {
                return { page: `${segments[segments.length - 2]}.html`, atSiteRoot: false };
            }
            return { page: last, atSiteRoot: false };
        }

        return { page: `${last}.html`, atSiteRoot: false };
    };

    const canonicalNavHref = (href) => {
        const normalized = String(href || '').trim().toLowerCase();
        if (!normalized || normalized === '/' || normalized === '/index.html') {
            return 'site-root';
        }

        const withoutLeadingSlash = normalized.startsWith('/') ? normalized.slice(1) : normalized;
        if (withoutLeadingSlash.endsWith('.html')) {
            const parts = withoutLeadingSlash.split('/').filter(Boolean);
            return parts[parts.length - 1] || 'index.html';
        }

        const trimmed = withoutLeadingSlash.replace(/\/+$/, '');
        if (!trimmed) return 'site-root';

        const parts = trimmed.split('/').filter(Boolean);
        if (!parts.length) return 'site-root';
        if (parts.length === 1) return 'index.html';
        return `${parts[parts.length - 1]}.html`;
    };

    const current = getCurrentPage();
    document.querySelectorAll('.site-nav-link').forEach((link) => {
        const hrefRaw = link.getAttribute('href') || '';
        const href = canonicalNavHref(hrefRaw);
        const isActive = href === 'site-root'
            ? current.atSiteRoot
            : (!current.atSiteRoot && href === current.page);

        if (isActive) {
            link.classList.add('active');
            link.setAttribute('aria-current', 'page');
        }
        link.addEventListener('click', closeNav);
    });
})();
