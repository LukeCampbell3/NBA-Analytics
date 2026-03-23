(() => {
    const body = document.body;
    const root = document.documentElement;
    const menuBtn = document.getElementById('siteMenuButton');
    const drawer = document.getElementById('siteNavDrawer');
    const overlay = document.getElementById('siteNavOverlay');
    const closeBtn = document.getElementById('siteNavClose');
    const themeToggleBtn = document.getElementById('themeToggleButton');
    const THEME_KEY = 'nba_theme_preference';

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

        if (!segments.length) return 'index.html';

        const last = segments[segments.length - 1];
        if (last.endsWith('.html')) {
            if (last === 'index.html' && segments.length > 1) {
                return `${segments[segments.length - 2]}.html`;
            }
            return last;
        }

        return `${last}.html`;
    };

    const current = getCurrentPage();
    document.querySelectorAll('.site-nav-link').forEach((link) => {
        const href = (link.getAttribute('href') || '').toLowerCase();
        if (href === current || (current === '' && href === 'index.html')) {
            link.classList.add('active');
            link.setAttribute('aria-current', 'page');
        }
        link.addEventListener('click', closeNav);
    });
})();
